> ℹ️ **Первым делом советую посмотреть презентации:**
>
> - [`research_notes/semester1.pdf`](https://github.com/oreshnya/grna_design_tool/blob/main/research_notes/semester1.pdf)
> - [`research_notes/semester2.pdf`](https://github.com/oreshnya/grna_design_tool/blob/main/research_notes/semester2.pdf)

# 📁 Структура проекта

### Отдельные файлы
- `run.py` — минимальный "чистый" запуск основных частей пайплайна (без блокнотов)  
- `RA_predictor_training.ipynb` — обучение модели *relative activity predictor*  
- `main_process.ipynb` — старый блокнот с процессом (функции перенесены в `modules/core.py`, вызовы — в `run.py`)  
- `ensemble.ipynb` — старое

---

### `data/`  
- `DAK1_short.txt`
- `DAK1.fasta`
- `emx1.hg38.targets.txt` — данные по off-targets, полученные из [CRISPRitz](https://github.com/pinellolab/CRISPRitz) по их тестовой последовательности gRNA на геноме человека  
- `ExampleDAK1seq.txt`  
- `Table_S8_machine_learning_input.csv` — данные для обучения *relative activity predictor*, источник: [Jost et al. (2020) *Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7065968/#SM1)  
- `genome_test_copy.fa`, `genome_test.fa` — тестовые фрагменты генома  

---

### `energy/`  
Модель для оценки энергии гибридизации РНК–ДНК, источник: [crisproff](https://github.com/RTH-tools/crisproff/tree/master)

---

### `models/`  
Модели, используемые в проекте  

- `CRISPR_BERT/` — источник: [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)  
- `Doench_2016/` — источник: [crispRdesignR](https://github.com/dylanbeeber/crispRdesignR)  
- `RA_predictor/` — собственная модель (структура и веса)

---

### `modules/`  
Почти все функции импортируются отсюда  

- `core.py` — основные функции, не разнесённые по модулям  
- `data_transformation.py` — обработка данных и схемы кодирования  
- `energy_calc.py` — модуль для расчёта энергии гибридизации ДНК–РНК  
- `model_evaluation.py` — функции для оценки моделей регрессии и классификации, построения кривых обучения  

---

### `research_notes/`  
Материалы с анализом, ссылками и отчётами  

- `ea & validation.md` — заметки по применению evolutionary algorithms для дизайна и оптимизации РНК  
- `performance_baseline.md` — предварительная оценка времени выполнения отдельных компонентов пайплайна  
- `semester1.pdf` — презентация-отчёт по практике за первый семестр  
- `semester2.pdf` — презентация-отчёт по практике за второй семестр  

---

### `R/`  
Код на **R**, используемый для вызова моделей Doench_2016 и обработки данных.  
Источник: [crispRdesignR](https://github.com/dylanbeeber/crispRdesignR)

---

### `SI/`  
Код reinforcement learning (A3C) для решения задачи *inverse folding*. Здесь уже кое-что подправлено - насколько я помню, основная проблема была в том, что в методе `create_layers` класса `RNAPolicy` один из слоев создавался с `activation_fn=None`.  

Источник: [P. Eastman et al. (2018) *Solving the RNA design problem with reinforcement learning*](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006176#sec016)

---

### `research_notes/`  
Материалы с анализом, ссылками и отчётами  

- `ea & validation.md` — заметки по применению evolutionary algorithms для дизайна и оптимизации РНК  
- `performance_baseline.md` — предварительная оценка времени выполнения отдельных компонентов пайплайна  
- `semester1.pdf` — презентация-отчёт по практике за первый семестр  
- `semester2.pdf` — презентация-отчёт по практике за второй семестр  


# Проблемы / зоны роста

CRISPR-BERT можно заменить на другую модель

Можно обратить внимание на [CCLMoff](https://github.com/duwa2/CCLMoff), статья [W. Du et. al. (2025) A versatile CRISPR/Cas9 system off-target prediction tool using language model](https://www.nature.com/articles/s42003-025-08275-6)

Авторы утверждают, что модель SOTA. Стоит проверить, на каких датасетах они сравнивались с другими моделями, и оценить скорость выполнения.


# ПО

- Python 3.10.11
- R 4.3.2

# R

Download R 4.3.2 from [official website](https://cran.r-project.org/bin/windows/base/old/)

Add environment variables:

    Path: C:\Program Files\R\R-4.3.2\bin

    R_HOME: C:\Program Files\R\R-4.3.2

Install needed packages with the folowing command:

```R
install.packages("package_name")
```

- gbm
- stringr
- vtreat
- BiocManager
- Biostrings

# Create virtual environment

Clone the repo:

```powershell
git clone https://github.com/oreshnya/grna_design_tool.git
```

Create virtual environment in project folder:

```bash
python -m venv venv
```

Activate virtual environment:

```bash 
. venv/Scripts/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

# Ubuntu 20.04 on Windows

На сайте https://store.rg-adguard.net/ вставить ссылку на Ubuntu из Microsoft Store, например:

```ruby
https://apps.microsoft.com/store/detail/ubuntu-2004-lts/9N6SVWS3RX71
```

и выбрать Retail

ПКМ на файл с расширением .appxbundle -> Копировать адрес ссылки

Скачать appxbundle через Git Bash:

```bash
curl -L "ВСТАВИТЬ ССЫЛКУ" -o ubuntu.appxbundle
```

В windows Powershell от имени администратора:

```powershell
Add-AppxPackage -Path "D:\path\to\ubuntu.appxbundle"
```
### Enable WSL

В powershell от имени администратора:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
```

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

### WSL 2

Установить WSL 2 kernel по [ссылке](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)

В powershell от имени администратора:

```powershell
wsl --set-default-version 2
```

# Genome test string

[UCSC](https://genome.ucsc.edu/cgi-bin/hgTables?hgsid=2632616484_CsPsFDwwAeaQyDpaOI3Zux7EqJjX&hgta_geneSeqType=genomic&hgta_doGenePredSequence=submit)