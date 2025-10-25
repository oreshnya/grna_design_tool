import os
import re
import io
from contextlib import redirect_stdout, redirect_stderr
import regex

import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.rinterface_lib.callbacks
import logging

from models.CRISPR_BERT.Encoder_change import BERT_encode, C_RNN_encode
from models.CRISPR_BERT.Encoder_change import Encoder
from models.CRISPR_BERT.model import build_bert

os.environ['RPY2_CCHAR_ENCODING'] = 'latin1'

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# utils = importr("utils") 
# utils.install_packages("gbm") # установка пакета R

pandas2ri.activate()
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

# Глобальный флаг, чтобы не инициализировать R много раз
_R_ENV_READY = False

def init_r_env():
    """Готовит окружение R: грузит модель и R-функции, если ещё не загружено."""
    global _R_ENV_READY
    if _R_ENV_READY:
        return  # уже всё сделано

    # ===== 1. посчитаем абсолютные пути =====
    # modules/core.py -> <repo_root>/modules/core.py
    current_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))

    r_script_path = os.path.join(repo_root, "R", "Doench_2016_Processing.R")
    r_script_path = os.path.abspath(r_script_path)

    r_model_path = os.path.join(repo_root, "models", "Doench_2016", "Rule_set_2_Model.rds")
    r_model_path = os.path.abspath(r_model_path)

    # ===== 2. загрузка модели в R =====
    model_path_r = r_model_path.replace("\\", "/")
    robjects.r(f'model <- readRDS("{model_path_r}")')

    # ===== 3. source R-скрипта =====
    base = importr('base')
    with redirect_stdout(io.StringIO()):
        base.source(r_script_path.replace("\\", "/"))

    _R_ENV_READY = True


def load_sequence(userseq: str) -> str:
    """Load sequence from FASTA, TXT, or direct string"""
    if userseq.endswith(".fasta") or userseq.endswith(".fa"):
        records = list(SeqIO.parse(userseq, "fasta"))
        sequence = str(records[0].seq)
    elif userseq.endswith(".txt"):
        with open(userseq, "r") as f:
            sequence = "".join(line.strip() for line in f.readlines())
    else:
        sequence = userseq.replace(" ", "").upper()
    return sequence


def find_candidate_sgRNAs(sequence: str, pam: str = "NGG") -> list[dict]:
    """Find candidate sgRNAs with PAM motif in both strands"""
    pam_regex = pam.replace("N", ".")
    results = []

    pam_pattern = re.compile(pam_regex)

    for strand, seq in [('+', sequence), ('-', str(Seq(sequence).reverse_complement()))]:
        for i in range(len(seq) - 29):
            window = seq[i:i+30]
            pam_candidate = window[24:24 + len(pam)]
            if pam_pattern.fullmatch(pam_candidate):
                sgRNA = window[4:24]
                results.append({
                    "strand": strand,
                    "start": i+4 if strand == "+" else len(seq) - i - 26,
                    "end": i+24 if strand == "+" else len(seq) - i - 6,
                    "sgRNA": sgRNA,
                    "pam": pam_candidate,
                    "window": window,

                })
    return results


DEGENERATE_BASES = re.compile(r"[UWSMKRYBDHVNZ]")  # некорректные символы
HOMOPOLYMER_PATTERN = re.compile(r"(A{4,}|T{4,}|G{4,}|C{4,})")

def is_valid(seq: str) -> bool:
    return not DEGENERATE_BASES.search(seq)

def get_gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)

def has_homopolymer(seq: str) -> bool:
    return bool(HOMOPOLYMER_PATTERN.search(seq))

BACKBONE = Seq("AGGCTAGTCCGT")
REVCOMP_BACKBONE = BACKBONE.reverse_complement()

def gc_content_4bp(seq4: str) -> float:
    return (seq4.count("G") + seq4.count("C")) / 4

def count_self_complementarity(sgRNA: str) -> int:
    sg_seq = Seq(sgRNA)
    revcomp_sg = sg_seq.reverse_complement()
    score = 0

    for i in range(0, len(sgRNA) - 3):  # 4-меры от 0 до 16
        tetramer = sgRNA[i:i+4]
        if gc_content_4bp(tetramer) >= 0.5:
            # Сравниваем с обратным комплементом самого sgRNA
            search_region = sgRNA[i+7:] if i <= 10 else ""
            if tetramer in Seq(search_region).reverse_complement():
                score += 1
            # Сравниваем с обратным комплементом бекбона
            if tetramer in REVCOMP_BACKBONE:
                score += 1
    return score

def filter_sgRNAs(candidates: list[dict]) -> list[dict]:
    filtered = []
    for c in candidates:
        if not is_valid(c["window"]):
            continue
        sgRNA_seq = c["sgRNA"]
        pam_seq = c["pam"]
        gc_content = get_gc_content(sgRNA_seq)
        homopolymer = has_homopolymer(sgRNA_seq)
        self_comp_score = count_self_complementarity(sgRNA_seq)

        filtered.append({
            **c,
            "sgRNA_seq": sgRNA_seq,
            "pam_seq": pam_seq,
            "gc_content": round(gc_content, 3),
            "homopolymer": homopolymer,
            "self_complementarity": self_comp_score
        })
    return filtered


def predict_from_r_model(df_features):
    gbm = importr("gbm")  # безопасный импорт пакета, не вызывает строковый вывод
    r_df = pandas2ri.py2rpy(df_features)
    robjects.globalenv["df_features_r"] = r_df

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pred = robjects.r('predict(model, df_features_r, n.trees=500)')

    return list(pred)


def build_dataframe(sgrna_candidates, features_df, efficiency_scores):
    """
    Сборка итогового датафрейма:
    - sgrna_candidates: список словарей с полями sgRNA, pam, strand, start, end, window
    - features_df: датафрейм признаков (выход Doench_2016_processing)
    - efficiency_scores: список float значений от модели
    """

    df = pd.DataFrame(sgrna_candidates)
    df["efficiency_score"] = [round(x, 3) for x in efficiency_scores]

    # Подсчёт GC-содержания 20-mer sgRNA
    df["GC_content"] = df["sgRNA"].apply(lambda x: round((x.count("G") + x.count("C")) / len(x), 3))

    # Детектирование гомополимеров
    df["Homopolymer"] = df["sgRNA"].apply(lambda x: any(rep in x for rep in ["AAAA", "TTTT", "GGGG", "CCCC"]))

    # Простейшая проверка самокомплементарности (временно — длина обратной комплементарной пары > 0)
    df["Self_complementary"] = df["sgRNA"].apply(lambda x: "CG" in x[::-1] or "GC" in x[::-1])  # упрощённо

    # Генерация Notes
    def generate_notes(row):
        notes = []
        if row["GC_content"] < 0.3:
            notes.append("Low GC")
        elif row["GC_content"] > 0.8:
            notes.append("High GC")
        if row["Homopolymer"]:
            notes.append("Homopolymer")
        if row["Self_complementary"]:
            notes.append("Self Complementary")
        if row["efficiency_score"] < 0.5:
            notes.append("Low Efficiency")
        if not notes:
            return "N/A"
        return ", ".join(notes)

    df["Notes"] = df.apply(generate_notes, axis=1)

    df = df.sort_values(by="efficiency_score", ascending=False).reset_index(drop=True)

    # import ace_tools as tools; tools.display_dataframe_to_user(name="sgRNA Candidates", dataframe=df)
    return df


def generate_features_in_r(sgrna_windows: list[str]):
    """Берёт список окон (20меров, 30меров и т.п.), вызывает R-функцию Doench_2016_processing,
    возвращает pandas.DataFrame с фичами для скоринга."""
    # гарантируем, что R окружение готово
    init_r_env()

    # конвертируем Python list[str] -> R StrVector
    sgrna_vector = robjects.StrVector(sgrna_windows)

    # вызываем ровно то имя, которое у тебя в R: Doench_2016_processing
    with redirect_stdout(io.StringIO()):
        r_df = robjects.r['Doench_2016_processing'](sgrna_vector)

    # R data.frame -> pandas.DataFrame
    return pandas2ri.rpy2py(r_df)


# ### Off-target screening

def _hamming(a: str, b: str) -> int:
    """Быстрое расстояние Хэмминга для одинаковой длины строк."""
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def _iupac_to_regex(pam: str) -> str:
    """грубый перевод IUPAC-кодов PAM в regex-паттерн."""
    iupac = {
        "A": "A", "C": "C", "G": "G", "T": "T",
        "R": "[AG]", "Y": "[CT]", "S": "[GC]", "W": "[AT]",
        "K": "[GT]", "M": "[AC]", "B": "[CGT]",
        "D": "[AGT]", "H": "[ACT]", "V": "[ACG]", "N": "[ACGT]",
    }
    return "".join(iupac.get(b, b) for b in pam.upper())


def off_target_screen_py_v2(
    sgrna_df: pd.DataFrame,
    genome_seq: str,
    userPAM: str = "NGG",
    PAM_col: str = "pam",
    sgRNA_seq_col: str = "sgRNA",
    strand_col: str = "strand",        # сохраняем, но не используем в расчёте
    start_col: str = "start",          # ─┐ нужны только
    end_col: str = "end",              # ─┘   если вам их важно вернуть
    gc_col: str = "GC_content",
    max_mismatches: int = 4,
):
    """
    Возвращает
    ----------
    sgrna_df_out : pd.DataFrame
        Исходный df + колонки MM0…MM4.
    off_targets  : pd.DataFrame
        Строки: потенциальные off-targets
        Индекс: тот же, что у исходной sgRNA-строки.
    """

    genome_seq = genome_seq.upper()
    pam_len = len(userPAM)

    sgrna_df["sgRNA_index"] = sgrna_df.index

    # «расширенный» список допустимых PAM'ов, если используется классическая Sp-Cas9-маска NGG
    if userPAM.upper() == "NGG":
        fwd_pams = {"GG", "AG", "CG", "GA", "GC", "GT", "TG"}   # + цепь
        rev_pams = {"CC", "CT", "CG", "TC", "GC", "AC", "CA"}   # – цепь
        pam_regex = _iupac_to_regex(userPAM)                    # == r"[ACGT]GG"
    else:
        fwd_pams = rev_pams = None
        pam_regex = _iupac_to_regex(userPAM)

    mm_cols_names = [f"MM{i}" for i in range(max_mismatches + 1)]
    mm_accumulator = {name: [] for name in mm_cols_names}
    off_target_rows = []

    # перебираем все sgRNA из датафрейма
    for idx, row in sgrna_df.iterrows():
        sg = row[sgRNA_seq_col].upper()

        # отбрасываем «дегенеративные» sgRNA
        if regex.search(r"[UWSMKRYBDHVNZ]", sg):
            for name in mm_cols_names:
                mm_accumulator[name].append(0)
            continue

        # паттерны для поиска с <=4 заменами
        core_sg = sg[:20]  # только 20 nt без PAM
        fwd_pat = regex.compile(f"({core_sg}){{s<={max_mismatches}}}", flags=regex.BESTMATCH)
        # fwd_pat = regex.compile(f"({sg}){{s<={max_mismatches}}}", flags=regex.BESTMATCH)
        rev_sg = str(Seq(sg).reverse_complement())
        rev_pat = regex.compile(f"({rev_sg}){{s<={max_mismatches}}}", flags=regex.BESTMATCH)

        # print(f"\n[DEBUG] Reverse sgRNA: {rev_sg}")

        counts = [0] * (max_mismatches + 1)   # счётчики MM0…MM4 для данной строки

        # ---------- поиск в прямой цепи ----------
        for m in fwd_pat.finditer(genome_seq, overlapped=True):
            start, end = m.span()                  # sgRNA-часть
            pam = genome_seq[end:end + pam_len]    # PAM сразу после неё

            if len(pam) < pam_len:
                continue
            if userPAM.upper() == "NGG" and pam[1:] not in fwd_pams:
                continue
            if not regex.fullmatch(pam_regex, pam):
                continue

            mism = _hamming(sg, genome_seq[start:end])
            counts[mism] += 1

            off_target_rows.append(
                {
                    "sgRNA_index": idx,
                    "sgRNA_seq": sg,
                    "sgRNA_PAM": row[PAM_col],
                    "off_target_seq": genome_seq[start:end],
                    "off_target_PAM": pam,
                    "strand": "+",
                    "start": start,
                    "end": end + pam_len,
                    "mismatches": mism,
                }
            )

        # ---------- поиск в обратной цепи ----------
        for m in rev_pat.finditer(genome_seq, overlapped=True):
            start, end = m.span()
            pam_start = start - pam_len
            if pam_start < 0:
                continue
            pam = genome_seq[pam_start:start]

            print(f"  Found reverse match: {genome_seq[start:end]} at {start}-{end}")
            print(f"  PAM before it: {pam}")

            if userPAM.upper() == "NGG" and pam[1:] not in rev_pams:
                continue
            if not regex.fullmatch(pam_regex, pam):
                continue

            mism = _hamming(rev_sg, genome_seq[start:end])
            counts[mism] += 1

            off_target_rows.append(
                {
                    "sgRNA_index": idx,
                    "sgRNA_seq": sg,
                    "sgRNA_PAM": row[PAM_col],
                    "off_target_seq": str(Seq(genome_seq[start:end]).reverse_complement()),
                    "off_target_PAM": str(Seq(pam).reverse_complement()),
                    "strand": "-",
                    "start": pam_start,
                    "end": end,
                    "mismatches": mism,
                }
            )

        # запоминаем «MM-колонки» для текущей sgRNA
        for i, name in enumerate(mm_cols_names):
            mm_accumulator[name].append(counts[i])

    # ---------- сборка результата ----------
    sgrna_df_out = sgrna_df.copy()
    for name in mm_cols_names:
        sgrna_df_out[name] = mm_accumulator[name]

    off_targets = pd.DataFrame(off_target_rows)
    if not off_targets.empty:
        off_targets = off_targets.sort_values(["sgRNA_index", "mismatches"])

    return sgrna_df_out, off_targets


# ### CRISPR-BERT

def load_off_targets_from_crispritz(path: str, n_rows: int = 1000) -> pd.DataFrame:
    """
    Грузим CRISPRitz TSV.
    Делаем:
    - sgRNA_without_PAM: гид без PAM (20 нуклеотидов)
    - off_target: офф-таргет (23 нт, uppercase)
    - sgRNA: гид с PAM (23 нт: 20 nt гида + PAM от офф-таргета)
    - label: временно -1
    """

    df_raw = pd.read_csv(path, sep=r"\t", engine="python", nrows=n_rows)

    required_cols = {"crRNA", "DNA"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"missing columns {missing}, got {list(df_raw.columns)}")

    def clean_crrna_take20(s: str) -> str:
        # CRISPRitz crRNA выглядит как 20nt + 'NNN'
        s = str(s).upper().replace("U", "T")
        if s.endswith("NNN") and len(s) >= 23:
            s = s[:-3]
        return s[:20]  # берём только 20 нт

    def clean_offtarget_full(s: str) -> str:
        # CRISPRitz DNA: офф-таргет, длина ~23, но с lowercase на мизматчах
        return str(s).upper().replace("U", "T")

    df = pd.DataFrame({
        "sgRNA_without_PAM": df_raw["crRNA"].map(clean_crrna_take20),
        "off_target": df_raw["DNA"].map(clean_offtarget_full),
    })

    # PAM = последние 3 нуклеотида off_target
    df["sgRNA"] = df["sgRNA_without_PAM"] + df["off_target"].str[-3:]

    # заглушка метки
    df["label"] = -1

    # sanity фильтр: должны быть ровно 23 символа у sgRNA/off_target
    df = df[
        (df["sgRNA_without_PAM"].str.len() == 20) &
        (df["sgRNA"].str.len() == 23) &
        (df["off_target"].str.len() == 23)
    ].reset_index(drop=True)

    return df[["sgRNA_without_PAM", "sgRNA", "off_target", "label"]]


def prepare_crisprbert_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает датафрейм для подачи в CRISPR-BERT:
    - совмещает sgRNA и off_target по позициям (одинаковой длины);
    - приводит пары нуклеотидов к нижнему регистру;
    - заменяет '_' на 'x';
    - соединяет в строку, разделённую запятой.
    """
    paired_sequences = []

    for idx, row in df.iterrows():
        sgrna = row["sgRNA"].replace("_", "X").lower()
        off = row["off_target"].replace("_", "X").lower()

        if len(sgrna) != len(off):
            print(f"⚠️ Строка #{idx}: длины sgRNA и off_target не совпадают → пропущена")
            paired_sequences.append("")  # или можно вставить заглушку 'xx,xx,...'
            continue

        pairs = [sgrna[i] + off[i] for i in range(len(sgrna))]
        sequence_str = ",".join(pairs)
        paired_sequences.append(sequence_str)

    df = df.copy()
    df["sequence"] = paired_sequences
    return df[["sequence", "label", "sgRNA", "off_target"]]


def predict_crisprbert_from_df(df: pd.DataFrame, model_weights_path: str, n_predict=100):
    """Кодирует и предсказывает off-target вероятность"""
    df_subset = df.iloc[:n_predict].copy()

    # BERT inputs
    data_list = df_subset[["sequence", "label"]].values.tolist()
    token_ids, segment_ids = BERT_encode(data_list)

    # RNN inputs
    rnn_encoded = np.array(C_RNN_encode(df_subset))

    # Модель
    model = build_bert()
    model.load_weights(model_weights_path)

    # Предсказания
    y_pred = model.predict([rnn_encoded, np.array(token_ids), np.array(segment_ids)])
    return df_subset, y_pred


def encode_sequence_column(df: pd.DataFrame, column: str = "sequence") -> pd.DataFrame:
    """
    Применяет Encoder к колонке с биграммами (в виде строки через запятую),
    добавляет закодированные данные в столбец 'rnn_encoded'.
    """
    encoded_list = []
    for i, val in enumerate(df[column]):
        try:
            merged = val.replace(",", "")  # склеиваем биграммы в строку длины 52
            en = Encoder(merged)
            encoded_list.append(en.on_off_code)
        except KeyError as e:
            print(f"⚠️ Ошибка кодирования в строке {i}: {e} → {val}")
            encoded_list.append(np.zeros((26, 7), dtype=int))  # fallback для некорректных строк

    df["rnn_encoded"] = encoded_list
    return df


def encode_with_bert_tokenizer(df: pd.DataFrame, column: str = "sequence"):
    """
    Применяет BERT_encode к колонке с биграммами и добавляет token_ids и segment_ids в датафрейм.
    """
    data_for_encoding = []
    for val in df[column]:
        tokens = val.split(',')
        tokens = [t.strip() for t in tokens if t]  # очистка
        text = " ".join(tokens)
        data_for_encoding.append((text, 0))  # фиктивный label

    token_ids, segment_ids = BERT_encode(data_for_encoding)
    df["bert_token_ids"] = token_ids
    df["bert_segment_ids"] = segment_ids
    return df


def add_rnn_encoded_column(df: pd.DataFrame, column: str = "sequence", n_nucl: int = 24) -> pd.DataFrame:
    """
    Добавляет столбец rnn_encoded — результат работы Encoder (OHE-кодировка биграмм).
    Если биграмм меньше 26, добавляется padding 'xx'.
    """
    encoded = []
    for i, val in enumerate(df[column]):
        try:
            bigrams = val.split(",")
            # паддинг до 26 биграмм
            if len(bigrams) < n_nucl:
                bigrams += ["xx"] * (n_nucl - len(bigrams))
            elif len(bigrams) > n_nucl:
                bigrams = bigrams[:n_nucl]
            # оборачиваем в строку без запятых
            merged = "".join(bigrams)
            en = Encoder(merged)
            encoded.append(en.on_off_code)
        except KeyError as e:
            print(f"⚠️ Ошибка кодирования RNN в строке {i}: {e} → {val}")
            encoded.append(np.zeros((n_nucl+2, 7)))  # с запасом
    df = df.copy()
    df["rnn_encoded"] = encoded
    return df


def add_bert_encoding_columns(df: pd.DataFrame, column: str = "sequence", n_nucl: int = 24) -> pd.DataFrame:
    """
    Добавляет в DataFrame две колонки:
    - bert_token_ids — токен-идентификаторы (из token_dict)
    - bert_segment_ids — список нулей того же размера.

    При необходимости добавляет 'xx'-паддинг до n_nucl биграмм.
    """
    data_for_encoding = []

    for i, val in enumerate(df[column]):
        try:
            bigrams = val.strip().split(",")
            # паддинг
            if len(bigrams) < n_nucl:
                bigrams += ["xx"] * (n_nucl - len(bigrams))
            elif len(bigrams) > n_nucl:
                bigrams = bigrams[:n_nucl]

            # преобразуем обратно в строку
            padded_text = ",".join(bigrams)
            data_for_encoding.append((padded_text, 0))  # фиктивный label
        except Exception as e:
            print(f"⚠️ Ошибка при подготовке строки #{i} для BERT: {e}")
            data_for_encoding.append((",".join(["xx"] * n_nucl), 0))

    token_ids, segment_ids = BERT_encode(data_for_encoding)

    df = df.copy()
    df["bert_token_ids"] = token_ids
    df["bert_segment_ids"] = segment_ids
    return df


def run_crisprbert_prediction(df: pd.DataFrame, model, n_predict: int = 100) -> pd.DataFrame:
    """
    Запускает CRISPR-BERT предсказания для первых n_predict строк датафрейма.
    Если n_predict = -1 → предсказываем для всех строк.
    Предполагает наличие 'rnn_encoded', 'bert_token_ids', 'bert_segment_ids'.
    Возвращает df с новой колонкой 'prediction' (вероятность off-target).
    Остальные строки получают "No pred".
    """
    df = df.copy()

    # если n_predict = -1, значит предсказываем для всех строк
    if n_predict == -1:
        n_predict = len(df)

    # подмножество для инференса
    df_subset = df.iloc[:n_predict]

    # собираем входы из первых n_predict строк
    rnn_encoded   = np.array(df_subset["rnn_encoded"].tolist())
    token_ids     = np.array(df_subset["bert_token_ids"].tolist())
    segment_ids   = np.array(df_subset["bert_segment_ids"].tolist())

    # модель даёт массив shape (n_predict, 2)
    y_pred = model.predict([rnn_encoded, token_ids, segment_ids])

    # Берём вероятность класса "on-target = 1" (index 1)
    probs = [float(p[1]) for p in y_pred]

    # создаём колонку и заполняем предсказаниями
    df["prediction"] = "No pred"
    # важно: используем .iloc для позиционной индексации
    df.iloc[:n_predict, df.columns.get_loc("prediction")] = probs

    return df