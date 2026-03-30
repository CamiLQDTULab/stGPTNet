# Đường dẫn file
file_path <- "D:/VietBao2025/Methods/Unsupervised_methods/Spado/SpaDo-main/DLPFC_all_slices.RData"

# Load dữ liệu
load(file_path)
slice_name <- "151507"

coords_151507 <- test_coordinate_list[[slice_name]]
coords_df <- data.frame(
  cell_id = rownames(coords_151507),
  X = coords_151507$X,
  Y = coords_151507$Y
)
write.csv(coords_df, "151507_coordinates.csv", row.names = FALSE)

coords_new <- read.csv("D:/VietBao2025/Methods/Supervised_Methods/151507_coordinates_h5ad.csv", stringsAsFactors = FALSE)

# đảm bảo cùng thứ tự
coords_new <- coords_new[match(rownames(coords_151507), coords_new$cell_id), ]

# update X, Y
coords_151507$X <- coords_new$X
coords_151507$Y <- coords_new$Y

test_coordinate_list[[slice_name]] <- coords_151507


# Gene expression
library(Matrix)

mat <- test_expression_list[[slice_name]]
mat_t <- t(mat)   # giờ là cell × gene
writeMM(mat_t, "matrix_151507.mtx")
# cell_id
write.csv(rownames(mat_t), "cell_ids.csv", row.names = FALSE)

# gene names
write.csv(colnames(mat_t), "gene_names.csv", row.names = FALSE)

library(Matrix)

# Tạo folder nếu chưa có
dir.create("matrix_files", showWarnings = FALSE)

# Lấy danh sách slice có sẵn
slice_names <- names(test_expression_list)

# Lọc các slice trong khoảng 151508 → 151676
slice_range <- slice_names[
  as.numeric(slice_names) >= 151508 & as.numeric(slice_names) <= 151676
]

# Loop qua từng slice
for (slice_name in slice_range) {
  
  cat("Đang xử lý slice:", slice_name, "\n")
  
  # Lấy matrix
  mat <- test_expression_list[[slice_name]]
  
  # transpose → cell × gene
  mat_t <- t(mat)
  
  # tạo tên file
  file_name <- paste0("matrix_files/matrix_", slice_name, ".mtx")
  
  # lưu file
  writeMM(mat_t, file_name)
}

cat("🎉 Hoàn thành tất cả slices!\n")


# update thông tin spatial từ dữ liệu h5ad vào dữ liệu Rdata
# Lấy danh sách slice có trong data
slice_names <- names(test_coordinate_list)

# Lọc các slice >= 151508
slice_range <- slice_names[as.numeric(slice_names) >= 151508]

for (slice_name in slice_range) {
  
  cat("🔄 Đang xử lý slice:", slice_name, "\n")
  
  # Lấy coords hiện tại
  coords <- test_coordinate_list[[slice_name]]
  
  # Đọc file CSV tương ứng

  file_path <- paste0("D:/VietBao2025/Methods/Unsupervised_methods/Spado/Transfer_h5data_2_Rdata/", 
                      slice_name, "_coordinates_h5ad.csv")
  
  if (!file.exists(file_path)) {
    cat("❌ Không tìm thấy file:", file_path, "\n")
    next
  }
  
  coords_new <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Align theo cell_id
  idx <- match(rownames(coords), coords_new$cell_id)
  
  # Check lỗi
  if (any(is.na(idx))) {
    cat("⚠️ Lỗi match cell_id ở slice:", slice_name, "\n")
    next
  }
  
  coords_new <- coords_new[idx, ]
  
  # Update X, Y
  coords$X <- coords_new$X
  coords$Y <- coords_new$Y
  
  # Gán lại vào list
  test_coordinate_list[[slice_name]] <- coords
  
  cat("✅ Hoàn thành slice:", slice_name, "\n\n")
}

cat("🎉 Update xong tất cả slice!\n")

# Lưu lại file Rdata đã updated
save(
  test_coordinate_list,
  sample_information_decon_list,
  sample_information_region_list,
  test_expression_list,
  file = "D:/VietBao2025/Methods/Unsupervised_methods/Spado/Transfer_h5data_2_Rdata/DLPFC_all_slices_updated.RData"
)
