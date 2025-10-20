## Структура проекта

data/ - данные
- DAK1_short.txt
- DAK1.fasta
- emx1.hg38.targets.txt - данные по off-targets, полученные из [CRISPRitz](https://github.com/pinellolab/CRISPRitz) по их тестовой последовательности gRNA на геноме человека
- ExampleDAK1seq.txt
- Table_S8_machine_learning_input.csv - данные для обучения relative activity predictor, источник: [Jost et. a. (2020) Titrating gene expression using libraries of systematically attenuated CRISPR guide RNAs](https://pmc.ncbi.nlm.nih.gov/articles/PMC7065968/) (см. Supplemantary materials)
- genome_test_copy.fa
- genome_test.fa


energy/
Модель для оценки энергии гибридизации РНК-ДНК, источник: [crisproff](https://github.com/RTH-tools/crisproff/tree/master)

models/
Модели, используемые в проекте

- CRISPR_BERT/ - источник: [CRISPR-BERT](https://github.com/BrokenStringx/CRISPR-BERT)
- Doench_2016/ - источник: [crispRdesignR](https://github.com/dylanbeeber/crispRdesignR)
- RA_predictor/ - наша структура и веса

modules/ 
Почти все функции импортируются отсюда

research_notes/
- ea & validation.md - изыскания насчет вариантов исопльзования EA для дизайна/оптимизации РНК и возможности валидации полцченных последжовательностей
- performance_baseline.md - предварительная оценка времени выоплнения отдельных компонентов планируемого пайплайна


## R

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

## Genome test string

[UCSC](https://genome.ucsc.edu/cgi-bin/hgTables?hgsid=2632616484_CsPsFDwwAeaQyDpaOI3Zux7EqJjX&hgta_geneSeqType=genomic&hgta_doGenePredSequence=submit)

## Reinvorcement Learning

Eastman et. al. (2018) [Solving the RNA design problem with reinforcement learning](https://doi.org/10.1371/journal.pcbi.1006176)

A3C (Asynchronous Advantage Actor Critic)

PPO (Proximal Policy Optimization), an architecture that improves our agent's training stability by avoiding too large policy updates.

A3C typically achieves quicker training times, but exhibits greater instability in reward values. Conversely, PPO demonstrates a more stable training process at the expense of longer execution times

## Ubuntu 20.04 on Windows

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