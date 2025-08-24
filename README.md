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