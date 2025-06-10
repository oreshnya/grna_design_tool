library(Biostrings)

off_target_screen <- function(sgrna_df, genomeseq, gtfname = NULL, userPAM = "NGG", calloffs = TRUE, annotateoffs = TRUE){
  requireNamespace("gbm", quietly = TRUE)
  if (missing(userPAM)) {
    userPAM <- "NGG"
  }
    {
    ## Check for off-targets in the genome
    ## Creates Function that converts all sgRNAs into a format readable
    ## by Biostrings
    setPAM <- userPAM

    sgRNA_seq <- sgrna_df$`sgRNA_seq`
    sgRNA_PAM <- sgrna_df$`pam`
    sgRNA_fow_or_rev <- sgrna_df$strand
    sgRNA_start <- sgrna_df$start
    sgRNA_end <- sgrna_df$end
    GCinstance <- sgrna_df$GC_content
    Homopolymerdetect <- sgrna_df$Homopolymer
    self_comp_list <- sgrna_df$Self_Complementary
    Efficiency_Score <- sgrna_df$Efficiency_Score
    # Notes <- sgrna_df$Notes
    sgRNA_with_PAM <- paste0(sgRNA_seq, sgRNA_PAM)
    lengthPAM <- nchar(setPAM)

    multiple_DNAString <- function(seqlist){
    Biostrings::DNAString(seqlist)
    }
    Biostrings_sgRNA <- lapply(sgRNA_with_PAM, multiple_DNAString)
    ## Define genome
    usegenome <- DNAStringSet(genomeseq)
    names(usegenome) <- "chr1"  # или любое имя
    seqnames <- names(usegenome)

    ## Removes any sgRNA that contain degerate bases
    sgRNA_seq <- sgRNA_seq[grepl("[UWSMKRYBDHVNZ]", sgRNA_seq) == FALSE]
    sgRNA_PAM <- sgRNA_PAM[grepl("[UWSMKRYBDHVNZ]", sgRNA_PAM) == FALSE]
    sgRNA_with_PAM <- sgRNA_with_PAM[grepl("[UWSMKRYBDHVNZ]", sgRNA_with_PAM) == FALSE]

    ## Creates a series of lists to store the incoming mismatch information
    mm0_list <- c()
    mm1_list <- c()
    mm2_list <- c()
    mm3_list <- c()
    mm4_list <- c()
    off_start <- c()
    off_end <- c()
    off_direction <- c()
    off_sgRNAseq <- c()
    off_offseq <- c()
    off_chr <- c()
    off_mismatch <- c()
    ## Creates a list of acceptable "NGG" PAMs
    PAM_test_list <- c("GG", "AG", "CG", "GA", "GC", "GT", "TG")
    rev_PAM_test_list <- c("CC", "CT", "CG", "TC", "GC", "AC", "CA")
    for (seqname in seqnames) {
    subject <- usegenome[[seqname]]
    message(paste("Checking for Off-Targets in", seqname, sep = " "))
    chrmm0_list <- c()
    chrmm1_list <- c()
    chrmm2_list <- c()
    chrmm3_list <- c()
    chrmm4_list <- c()
    revchrmm0_list <- c()
    revchrmm1_list <- c()
    revchrmm2_list <- c()
    revchrmm3_list <- c()
    revchrmm4_list <- c()
    for (pattern in Biostrings_sgRNA) {
        usepattern <- Biostrings::DNAString(substr(as.character(pattern), 1, 20))
        ## Searches for off-targets in the forward strand
        off_info <- Biostrings::matchPattern(usepattern, subject, max.mismatch = 4, min.mismatch = 0, fixed = TRUE)
        if (length(off_info) > 0) {
        off_info_full <- IRanges::Views(subject, BiocGenerics::start(off_info), BiocGenerics::end(off_info)+lengthPAM)
        if (setPAM == "NGG") {
            off_info_position <- which(substr(as.character(off_info_full), 22, 23) %in% PAM_test_list)
            off_info <- off_info[off_info_position]
            off_info_full <- off_info_full[off_info_position]
        } else {
            off_info_position <- which(stringr::str_detect(substr(as.character(off_info_full), 21, 20+lengthPAM), usesetPAM))
            off_info <- off_info[off_info_position]
            off_info_full <- off_info_full[off_info_position]
        }
        }
        mis_info <- IRanges::elementNROWS(Biostrings::mismatch(usepattern, off_info))
        if (length(off_info) > 0) {
        seqs_w_4mm <- which(mis_info == 4)
        seqs_w_off_PAM <- which(substr(as.character(off_info_full), 22, 23) %in% c("AG", "CG", "GA", "GC", "GT", "TG"))
        discard_offs <- intersect(seqs_w_4mm, seqs_w_off_PAM)
        if (length(discard_offs) != 0) {
            off_info <- off_info[-discard_offs]
            off_info_full <- off_info_full[-discard_offs]
            mis_info <- mis_info[-discard_offs]
        }
        }
        ## Searches for off-targets in the reverse strand
        rev_pattern <- Biostrings::reverseComplement(usepattern)
        ### rev_off_info <- Biostrings::matchPattern(rev_pattern, subject, max.mismatch = 4, min.mismatch = 0, fixed = c(pattern = FALSE, subject = TRUE))
        rev_off_info <- Biostrings::matchPattern(rev_pattern, subject, max.mismatch = 4, min.mismatch = 0, fixed = TRUE)
        if (length(rev_off_info) > 0) {
        rev_off_info_full <- IRanges::Views(subject, (BiocGenerics::start(rev_off_info)-lengthPAM), BiocGenerics::end(rev_off_info))
        if (setPAM == "NGG") {
            rev_off_info_position <- which(substr(as.character(rev_off_info_full), 1, 2) %in% rev_PAM_test_list)
            rev_off_info <- rev_off_info[rev_off_info_position]
            rev_off_info_full <- rev_off_info_full[rev_off_info_position]
        } else {
            rev_off_info_position <- which(stringr::str_detect(substr(as.character(rev_off_info_full), 1, lengthPAM), revusesetPAM))
            rev_off_info <- rev_off_info[rev_off_info_position]
            rev_off_info_full <- rev_off_info_full[rev_off_info_position]
        }
        }
        rev_mis_info <- IRanges::elementNROWS(Biostrings::mismatch(rev_pattern, rev_off_info))
        if (length(rev_mis_info) > 0) {
        rev_seqs_w_4mm <- which(rev_mis_info == 4)
        rev_seqs_w_off_PAM <- which(substr(as.character(rev_off_info_full), 1, 2) %in% c("CT", "CG", "TC", "GC", "AC", "CA"))
        rev_discard_offs <- intersect(rev_seqs_w_4mm, rev_seqs_w_off_PAM)
        if (length(rev_discard_offs) != 0) {
            rev_off_info <- rev_off_info[-rev_discard_offs]
            rev_off_info_full <- rev_off_info_full[-rev_discard_offs]
            rev_mis_info <- rev_mis_info[-rev_discard_offs]
        }
        }
        if (length(off_info) > 0) {
        for (f in 1:length(off_info)) {
            off_start[length(off_start)+1] <- BiocGenerics::start(off_info)[f]
            off_end[length(off_end)+1] <- BiocGenerics::end(off_info)[f]+lengthPAM
            off_direction[length(off_direction)+1] <- "+"
            off_chr[length(off_chr)+1] <- seqname
            off_mismatch[length(off_mismatch)+1] <- mis_info[f]
            off_sgRNAseq[length(off_sgRNAseq)+1] <- as.character(pattern)
            off_offseq[length(off_offseq)+1] <- as.character(off_info_full[f])
        }
        }
        if (length(rev_off_info) > 0) {
        for (f in 1:length(rev_off_info)) {
            off_start[length(off_start)+1] <- BiocGenerics::start(rev_off_info)[f]-lengthPAM
            off_end[length(off_end)+1] <- BiocGenerics::end(rev_off_info)[f]
            off_direction[length(off_direction)+1] <- "-"
            off_chr[length(off_chr)+1] <- seqname
            off_mismatch[length(off_mismatch)+1] <- rev_mis_info[f]
            off_sgRNAseq[length(off_sgRNAseq)+1] <- as.character(pattern)
            off_offseq[length(off_offseq)+1] <- as.character(rev_off_info_full[f])
        }
        }
        individMM <- c()
        if (length(mis_info) > 0) {
        for (f in 1:length(mis_info)) {
            individMM[length(individMM)+1] <- mis_info[f]
        }
        chrmm0_list[length(chrmm0_list)+1] <- sum(individMM == 0)
        chrmm1_list[length(chrmm1_list)+1] <- sum(individMM == 1)
        chrmm2_list[length(chrmm2_list)+1] <- sum(individMM == 2)
        chrmm3_list[length(chrmm3_list)+1] <- sum(individMM == 3)
        chrmm4_list[length(chrmm4_list)+1] <- sum(individMM == 4)
        } else {
        chrmm0_list[length(chrmm0_list)+1] <- 0
        chrmm1_list[length(chrmm1_list)+1] <- 0
        chrmm2_list[length(chrmm2_list)+1] <- 0
        chrmm3_list[length(chrmm3_list)+1] <- 0
        chrmm4_list[length(chrmm4_list)+1] <- 0
        }
        individMM <- c()
        if (length(rev_mis_info) > 0) {
        for (f in 1:length(rev_mis_info)) {
            individMM[length(individMM)+1] <- rev_mis_info[f]
        }
        revchrmm0_list[length(revchrmm0_list)+1] <- sum(individMM == 0)
        revchrmm1_list[length(revchrmm1_list)+1] <- sum(individMM == 1)
        revchrmm2_list[length(revchrmm2_list)+1] <- sum(individMM == 2)
        revchrmm3_list[length(revchrmm3_list)+1] <- sum(individMM == 3)
        revchrmm4_list[length(revchrmm4_list)+1] <- sum(individMM == 4)
        } else {
        revchrmm0_list[length(revchrmm0_list)+1] <- 0
        revchrmm1_list[length(revchrmm1_list)+1] <- 0
        revchrmm2_list[length(revchrmm2_list)+1] <- 0
        revchrmm3_list[length(revchrmm3_list)+1] <- 0
        revchrmm4_list[length(revchrmm4_list)+1] <- 0
        }
    }
    if (is.null(mm0_list)) {
        mm0_list <- chrmm0_list + revchrmm0_list
        mm1_list <- chrmm1_list + revchrmm1_list
        mm2_list <- chrmm2_list + revchrmm2_list
        mm3_list <- chrmm3_list + revchrmm3_list
        mm4_list <- chrmm4_list + revchrmm4_list
    } else {
        mm0_list <- chrmm0_list + mm0_list + revchrmm0_list
        mm1_list <- chrmm1_list + mm1_list + revchrmm1_list
        mm2_list <- chrmm2_list + mm2_list + revchrmm2_list
        mm3_list <- chrmm3_list + mm3_list + revchrmm3_list
        mm4_list <- chrmm4_list + mm4_list + revchrmm4_list
    }

    # Пустой датафрейм-заглушка для off-target info
    all_offtarget_info <- data.frame("NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA")
    colnames(all_offtarget_info) <- c(
      "sgRNA sequence", "Chromosome", "Start", "End", "Mismatches", "Direction",
      "CFD Score", "Off-target sequence", "Gene ID", "Gene Name", "Sequence Type", "Exon Number"
    )

    # Финальный датафрейм по sgRNA
    sgRNA_data <- data.frame(
      sgRNA_seq, sgRNA_PAM, sgRNA_fow_or_rev, sgRNA_start, sgRNA_end,
      GCinstance, Homopolymerdetect, self_comp_list, Efficiency_Score,
      mm0_list, mm1_list, mm2_list, mm3_list, mm4_list
    #   , Notes
    )
    colnames(sgRNA_data) <- c(
      "sgRNA sequence", "PAM sequence", "Direction", "Start", "End",
      "GC content", "Homopolymer", "Self Complementary", "Efficiency Score",
      "MM0", "MM1", "MM2", "MM3", "MM4"
    #   ,"Notes"
    )

    sgRNA_data <- sgRNA_data[order(-sgRNA_data$`Efficiency Score`),]

    # Вернуть список как NamedList
    return(list(sgRNA_data = sgRNA_data, all_offtarget_info = all_offtarget_info))
}

}
#     ## Calculates off-target scores for each off target sequence
#     message("Annotating off-targets")
#     off_model_PAMs <- c("AG", "CG", "GA", "GC", "GT", "TG")
#     CFD_PAM_Scores <- data.frame(off_model_PAMs, c(0.259259, 0.107142, 0.069444, 0.022222, 0.016129, 0.038961))
#     CFD_Scores <- c()
#     for (x in 1:length(off_offseq)) {
#     if (off_direction[x] == "-") {
#         temporary_off <- Biostrings::DNAString(off_offseq[x])
#         temporary_off <- Biostrings::reverseComplement(temporary_off)
#         CFDoffsplit <- stringr::str_split(as.character(temporary_off), "", simplify = TRUE)
#     } else {
#         CFDoffsplit <- stringr::str_split(off_offseq[x], "", simplify = TRUE)
#     }
#     CFDsgRNAsplit <- stringr::str_split(off_sgRNAseq[x], "", simplify = TRUE)
#     individ_scores <- c()
#     for (g in 1:20) {
#         if (CFDsgRNAsplit[g] != CFDoffsplit[g]) {
#         index <- which(CFD_Model_Scores$Position==g & CFD_Model_Scores$sgRNA==CFDsgRNAsplit[g] & CFD_Model_Scores$DNA==CFDoffsplit[g])
#         individ_scores[length(individ_scores)+1] <- CFD_Model_Scores[index,4]
#         }
#     }
#     if (setPAM == "NGG") {
#         specific_PAM <- (paste(CFDoffsplit[22], CFDoffsplit[23], sep = ""))
#         if (isTRUE(specific_PAM != "GG")){
#         if (specific_PAM %in% off_model_PAMs) {
#             PAM_index <- which(off_model_PAMs==specific_PAM)
#             individ_scores[length(individ_scores)+1] <- CFD_PAM_Scores[PAM_index,2]
#         } else {
#             individ_scores[length(individ_scores)+1] <- 0
#         }
#         }
#     }
#     if (length(individ_scores) == 0) {
#         CFD_Scores[length(CFD_Scores)+1] <- 1
#     } else {
#         CFDproduct <- 1
#         for (x in 1:length(individ_scores)){
#         CFDproduct <- prod(individ_scores[x], CFDproduct)
#         }
#         CFD_Scores[length(CFD_Scores)+1] <- CFDproduct
#     }
#     }
#     CFD_Scores <- round(CFD_Scores, digits = 3)
#     ## Decides whether to annotate off_targets
#     if (((sum(mm0_list) + sum(mm1_list) + sum(mm2_list) + sum(mm3_list)) == 0) || (annotateoffs == FALSE)) {
#     # Temporary fix to maintain the structure of the output data frame
#     self_comp_list <- unlist(self_comp_list)
#     ## Put lists in data frame
#     sgRNA_data <- data.frame(sgRNA_seq, sgRNA_PAM, sgRNA_fow_or_rev, sgRNA_start, sgRNA_end, GCinstance, Homopolymerdetect, self_comp_list, Efficiency_Score, mm0_list, mm1_list, mm2_list, mm3_list, mm4_list, Notes)
#     ## Set the names of each column
#     colnames(sgRNA_data) <- c("sgRNA sequence", "PAM sequence", "Direction", "Start", "End", "GC content", "Homopolymer", "Self Complementary", "Efficiency Score", "MM0", "MM1", "MM2", "MM3", "MM4", "Notes")
#     sgRNA_data <- sgRNA_data[order(-sgRNA_data$`Efficiency Score`),]
#     all_offtarget_info <- data.frame(off_sgRNAseq, off_chr, off_start, off_end, off_mismatch, off_direction, CFD_Scores, off_offseq, "NA", "NA", "NA", "NA")
#     colnames(all_offtarget_info) <- c("sgRNA sequence", "Chromosome", "Start", "End", "Mismatches", "Direction", "CFD Score", "Off-target sequence", "Gene ID", "Gene Name", "Sequence Type", "Exon Number")
#     data_list <- c("sgRNA_data" = sgRNA_data, "all_offtarget_info" = all_offtarget_info)
#     data_list
#     } else {
#     ## Creates a function that annotates the off-targets called above
#     annotate_genome <- function(ochr, ostart, oend, odir, gtfname) {
#         gtf <- rtracklayer::import(gtfname)
#         GenomeInfoDb::seqlevelsStyle(gtf) <- "UCSC"
#         seqer <- unlist(ochr)
#         starter <- as.numeric(ostart)
#         ender <- as.numeric(unlist(oend))
#         strander <- unlist(odir)
#         off_ranges <- GenomicRanges::GRanges(seqer, IRanges::IRanges(starter, ender), strander)
#         olaps <- IRanges::findOverlaps(off_ranges, gtf)
#         geneid <- c()
#         geneidlist <- c()
#         genename <- c()
#         genenamelist <- c()
#         sequencetype <- c()
#         sequencetypelist <- c()
#         exonnumber <- c()
#         exonnumberlist <- c()
#         S4Vectors::mcols(off_ranges)$gene_id <- c()
#         for (p in 1:length(off_ranges)) {
#         if (p %in% S4Vectors::queryHits(olaps)) {
#             geneid <- S4Vectors::mcols(gtf)$gene_id[S4Vectors::subjectHits(olaps[which(p == S4Vectors::queryHits(olaps))])]
#             geneid <- unique(geneid)
#             geneidlist[length(geneidlist)+1] <- paste(geneid, collapse = ", ")
#             genename <- S4Vectors::mcols(gtf)$gene_name[S4Vectors::subjectHits(olaps[which(p == S4Vectors::queryHits(olaps))])]
#             genename <- unique(genename)
#             genenamelist[length(genenamelist)+1] <- paste(genename, collapse = ", ")
#             sequencetype <- S4Vectors::mcols(gtf)$type[S4Vectors::subjectHits(olaps[which(p == S4Vectors::queryHits(olaps))])]
#             sequencetype <- unique(sequencetype)
#             sequencetypelist[length(sequencetypelist)+1] <- paste(sequencetype, collapse = ", ")
#             exonnumber <- S4Vectors::mcols(gtf)$exon_number[S4Vectors::subjectHits(olaps[which(p == S4Vectors::queryHits(olaps))])]
#             exonnumber <- unique(exonnumber)
#             exonnumber <- exonnumber[-which(is.na(exonnumber))]
#             exonnumberlist[length(exonnumberlist)+1] <- paste(exonnumber, collapse = ", ")
#         } else {
#             geneidlist[length(geneidlist)+1] <- "NA"
#             genenamelist[length(genenamelist)+1] <- "NA"
#             sequencetypelist[length(sequencetypelist)+1] <- "NA"
#             exonnumberlist[length(exonnumberlist)+1] <- "NA"
#         }
#         }
#         S4Vectors::mcols(off_ranges)$gene_id <- geneidlist
#         more_off_info <- data.frame(geneidlist, genenamelist, sequencetypelist, exonnumberlist)
#         more_off_info
#     }
#     ## Ensures that all off-targets provided run in the same direction as the sgRNA sequence
#     if (("-" %in% off_direction) == TRUE) {
#         revcomp_index <- which(off_direction == "-")
#         to_be_revcomped <- c(off_offseq[revcomp_index])
#         new_offs <- c()
#         x <- 1
#         for (x in 1:length(to_be_revcomped)) {
#         new_offs[length(new_offs)+1] <- as.character(Biostrings::reverseComplement(Biostrings::DNAString(to_be_revcomped[x])))
#         }
#         for (x in 1:length(revcomp_index)) {
#         off_offseq[revcomp_index[x]] <- new_offs[x]
#         }
#     }
#     ## Compiles data frame of all off-target annotations
#     more_off_info <- annotate_genome(off_chr, off_start, off_end, off_direction, gtfname)
#     ## Complies all extra sgRNA info into a separate data frame
#     all_offtarget_info <- data.frame(off_sgRNAseq, off_chr, off_start, off_end, off_mismatch, off_direction, CFD_Scores, off_offseq, more_off_info$geneidlist, more_off_info$genenamelist, more_off_info$sequencetypelist, more_off_info$exonnumberlist)
#     colnames(all_offtarget_info) <- c("sgRNA sequence", "Chromosome", "Start", "End", "Mismatches", "Direction", "CFD Scores", "Off-target sequence", "Gene ID", "Gene Name", "Sequence Type", "Exon Number")
#     # Temporary fix to maintain the structure of the output data frame
#     self_comp_list <- unlist(self_comp_list)
#     ## Put lists in data frame
#     sgRNA_data <- data.frame(sgRNA_seq, sgRNA_PAM, sgRNA_fow_or_rev, sgRNA_start, sgRNA_end, GCinstance, Homopolymerdetect, self_comp_list, Efficiency_Score, mm0_list, mm1_list, mm2_list, mm3_list, mm4_list, Notes)
#     ## Set the names of each column
#     colnames(sgRNA_data) <- c("sgRNA sequence", "PAM sequence", "Direction", "Start", "End", "GC content", "Homopolymer", "Self Complementary", "Efficiency Score", "MM0", "MM1", "MM2", "MM3", "MM4", "Notes")
#     sgRNA_data <- sgRNA_data[order(-sgRNA_data$`Efficiency Score`),]
#     data_list <- c("sgRNA_data" = sgRNA_data, "all_offtarget_info" = all_offtarget_info)
#     data_list
#     }
# }
# } else {
# data_list <- data.frame()
}