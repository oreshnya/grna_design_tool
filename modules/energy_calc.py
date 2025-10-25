import pickle, os

"""
energy_calc.py — модуль для расчёта энергии гибридизации ДНК–РНК.

Содержит функции для подготовки и анализа пар sgRNA ↔ off-target:
- init_energy_tables() — лениво загружает предвычисленные термодинамические таблицы (energy/energy_dics.pkl);
- add_otSeq_column(df) — добавляет в датафрейм комплементарную последовательность (otSeq) по столбцу sgRNA_seq;
- compute_hybridization_energy(df) — вычисляет суммарную энергию гибридизации ΔGₕ для каждой пары sgRNA_seq / otSeq;
- calcRNADNAenergy() — низкоуровневая функция, рассчитывающая вклад пар оснований и петель по таблицам энергий.

Использует словари соответствий нуклеотидов (RI_REV_NT_MAP), допустимых пар (RI_MATCH_noGU)
и внутренние поправки (RNA_DNA_internal_loop).
"""

# === Константы и таблицы, которые уже известны заранее ===

RI_REV_NT_MAP = {
    '-':'', 'a':'T', 'A':'T', 'c':'G', 'C':'G', 'g':'C', 'G':'C',
    't':'A', 'T':'A', 'u':'A', 'U':'A', 'n':'N', 'N':'N'
}

RNA_DNA_internal_loop = {
    3: 3.2, 4: 3.555, 5: 3.725, 6: 3.975,
    7: 4.16, 8: 4.33, 9: 4.495, 10: 4.6, 11: 4.7
}

RI_MATCH_noGU = {'A':{'A':False, 'C':False, 'G':False, 'T':True},
         'C':{'A':False, 'C':False, 'G':True, 'T':False},
         'G':{'A':False, 'C':True, 'G':False, 'T':False},
         'T':{'A':True, 'C':False, 'G':False, 'T':False}}

# === ЛЕНИВО загружаемые таблицы энергии ===

RNA_DNA = None 

def init_energy_tables():
    """
    Лениво загружает таблицы энергий из energy/energy_dics.pkl в глобальную переменную RNA_DNA.
    Если уже загружено — просто возвращается.
    """
    global RNA_DNA

    if RNA_DNA is not None:
        return  # уже инициализировано

    # путь до energy/energy_dics.pkl относительно этого файла
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    energy_path = os.path.join(repo_root, "energy", "energy_dics.pkl")

    with open(energy_path, "rb") as f:
        loaded = pickle.load(f)

    # твоя логика из ноутбука: берём только ключи 0 и 1
    RNA_DNA = {
        0: loaded[0],
        1: loaded[1],
    }


def add_otSeq_column(df, sgrna_col: str = "sgRNA_seq", otseq_col: str = "otSeq"):
    """
    Добавляет колонку otSeq = реверс-комплемент sgRNA (как ДНК->РНК-партнёр).
    """
    def reverse_complement(dna):
        return ''.join(RI_REV_NT_MAP.get(base, base) for base in reversed(dna.upper()))

    df[otseq_col] = df[sgrna_col].apply(reverse_complement)
    return df


def compute_hybridization_energy(df, sgrna_col: str = "sgRNA_seq", otseq_col: str = "otSeq", energy_col: str = "deltaG_h"):
    """
    Для каждой строки считает суммарную энергию гибридизации sgRNA vs otSeq.
    Записывает значения в energy_col (по умолчанию deltaG_h).
    """
    init_energy_tables()

    def compute_energy(row):
        try:
            guide_seq = row[sgrna_col].upper()
            ot_seq = row[otseq_col]
            return sum(calcRNADNAenergy(guide_seq, ot_seq))
        except Exception:
            return None

    df[energy_col] = df.apply(compute_energy, axis=1)
    return df


def calcRNADNAenergy(guideSeq: str, otSeq: str, GU_allowed: bool = False):
    """
    Возвращает список энергий по позициям, затем мы делаем sum() снаружи.
    """
    # guideSeq = guideSeq.upper()[:-3]
    # seq = ''.join([RI_REV_NT_MAP[c] for c in otSeq[:-3]])
    guideSeq = guideSeq.upper()
    seq = ''.join([RI_REV_NT_MAP[c] for c in otSeq])

    spos = -1
    epos = -1

    energy = [0.0]*len(guideSeq)

    MATCH = RI_MATCH_noGU

    for i in range(len(seq)):
        if MATCH[guideSeq[i]][seq[i]]:
            if spos==-1:
                spos = i
            epos = i

    i = spos
    while i<epos:
        j = i+1
        while MATCH[seq[j]][guideSeq[j]]==False:
            j = j+1

            if j > epos:
                break
        if j > epos:
            break

        loop_size = (j-i)-1
        eng_con = 0
        if loop_size < 3:
            key_guide = guideSeq[i:i+2]
            key_seq = seq[i:i+2]
            try:
                eng_con = RNA_DNA[loop_size][key_guide][key_seq]
            except KeyError:
                eng_con = 0  # если пары нет в таблице, считаем ΔG = 0

            # if there is a stack in the beginning or end AU GU penalty is still needed
            if loop_size == 0:
                if (i==spos and (guideSeq[i]=="T" or seq[i]=="T")) or (j==epos and (guideSeq[j]=="T" or seq[j]=="T")):
                    eng_con += 0.25
        else:
            eng_con = float(RNA_DNA_internal_loop[loop_size]) + float(RNA_DNA[0][guideSeq[i:i+2]][seq[i:i+2]]) + float(RNA_DNA[0][guideSeq[j-1:j+1]][seq[j-1:j+1]])

        for k in range(loop_size+1):
            energy[i+k] += eng_con/(loop_size+1)

        i = j

    return energy