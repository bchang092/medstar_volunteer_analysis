# install.packages(c("readxl","dplyr","stringr","tidyr","ggplot2","gridExtra","reticulate"))
library(readxl)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(gridExtra)
install.packages("reticulate")
library(reticulate)
reticulate::py_install(c("bertopic","sentence-transformers"), pip = TRUE)


# 1. Load your data
setwd("/Users/brandonchang/Desktop/rounding_survey/")
file_path <- "06152025_data.xlsx"  # adjust path if needed
df <- read_excel(file_path)

# 2. Rename all the "multi-item" question columns
orig <- names(df)

# Q1 = "What requests did patients make? … .<Item>"
q1_pat <- "^What requests did patients make\\?"
q1_cols <- grep(q1_pat, orig, value = TRUE)

# Q2 = "What percentage of patients feel like they waited … .<Service>"
q2_pat <- "^What percentage of patients feel like they waited"
q2_cols <- grep(q2_pat, orig, value = TRUE)

# strip off everything through the last dot + any spaces
new_q1 <- str_replace(q1_cols, ".*\\.\\s*", "")
new_q2 <- str_replace(q2_cols, ".*\\.\\s*", "")

names(df)[match(q1_cols, orig)] <- new_q1
names(df)[match(q2_cols, orig)] <- new_q2

# 3. Normalize all numeric form‐entries by "Shift Floor"
shift_floor_col <- "Shift Floor"   # change if your column name differs
num_cols <- names(df)[sapply(df, is.numeric) & names(df) != shift_floor_col]
# coerce Shift Floor to numeric (if it isn’t already)
df[[shift_floor_col]] <- as.numeric(as.character(df[[shift_floor_col]]))

df[num_cols] <- df[num_cols] / df[[shift_floor_col]]

# 4. Set up BERTopic via reticulate
# reticulate::py_install(c("bertopic","sentence-transformers"), pip = TRUE)
bt <- import("bertopic")
cv <- import("sklearn.feature_extraction.text")$CountVectorizer
embed_mod <- import("sentence_transformers")$SentenceTransformer("all-MiniLM-L6-v2")
topic_model <- bt$BERTopic(embedding_model = embed_mod, vectorizer_model = cv())

# Identify your free‐text columns
pos_col <- grep("Select the most positive event", names(df), value = TRUE)
neg_col <- grep("Select the most negative event", names(df), value = TRUE)

# 5. Loop over departments and plot
group_col <- "Department"    # ← change to whatever column denotes your department
pdf("department_plots.pdf", width = 8, height = 10)

for(dept in unique(df[[group_col]])) {
  d <- df %>% filter(.data[[group_col]] == dept)
  
  # — Q1: requests per shift (stacked by item)
  d1 <- d %>% select(all_of(new_q1)) %>%
    pivot_longer(everything(), names_to = "Request", values_to = "Qty")
  p1 <- ggplot(d1, aes(x = Request, y = Qty, fill = Request)) +
    geom_bar(stat = "identity") +
    labs(title = paste(dept, "– Patient requests"),
         y = "Quantity / Shift") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")
  
  # — Q2: wait‐% per service
  d2 <- d %>% select(all_of(new_q2)) %>%
    pivot_longer(everything(), names_to = "Service", values_to = "WaitPct")
  p2 <- ggplot(d2, aes(x = Service, y = WaitPct, fill = Service)) +
    geom_bar(stat = "identity") +
    labs(title = paste(dept, "– Wait time (%)"),
         y = "Percentage") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")
  
  # — Q3: BERTopic themes
  texts <- c(d[[pos_col]], d[[neg_col]])
  texts <- texts[!is.na(texts) & texts != ""]
  res <- topic_model$fit_transform(texts)
  topics <- as.integer(res[[1]])
  info_py <- topic_model$get_topic_info()
  info_df <- py_to_r(info_py)
  topn <- head(info_df, 10)
  topn$Frequency <- as.numeric(topn$Count)
  p3 <- ggplot(topn, aes(x = reorder(Name, Frequency), y = Frequency)) +
    geom_col() +
    coord_flip() +
    labs(title = paste(dept, "– Top BERTopic themes"),
         y = "Frequency", x = "") +
    theme_minimal()
  
  # arrange all three on one page
  grid.arrange(p1, p2, p3, ncol = 1)
  grid.newpage()
}

dev.off()
