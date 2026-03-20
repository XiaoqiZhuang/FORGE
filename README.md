# FORGE
Canopy Delination Detection for drone images of the Amazon forest.


python yolo_inference.py  --img_path Z:/mnt/parscratch/users/acr23xz/Forge/raw_data/2023-21/17-MAD-TAHPER-FMP-2021-041_08112023_004_03112023_transparent_mosaic_group1.tif  --model_path D:/Forge/runs_baseline/baseline_real_data/weights/best.pt --conf_threshold 0.3  --output_dir ../data/  


python crown_retrieval.py  --tif_path Z:/mnt/parscratch/users/acr23xz/Forge/raw_data/2023-21/17-MAD-TAHPER-FMP-2021-041_08112023_004_03112023_transparent_mosaic_group1.tif  --candidate_pool ../stage1_segmentation/results/17-MAD-TAHPER-FMP-2021-041_08112023_004_03112023_transparent_mosaic_group1_predictions.gpkg  --output_dir ../data/ 