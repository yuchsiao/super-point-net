for i in `seq 0 4`;
do
    python augment_data.py data/modelnet40_ply_hdf5_2048/ply_data_train$i.h5 data/augmented/train_data_$3_$4_$1_$2_$i.h5 \
    --method $1 --normalize-weight $2 --input-num-points $3 --num-clusters $4
    echo "data/augmented/train_data_$3_$4_$1_$2_$i.h5" >> data/augmented/train_files_$3_$4_$1_$2.txt
done

for i in `seq 0 1`;
do
    python augment_data.py data/modelnet40_ply_hdf5_2048/ply_data_test$i.h5 data/augmented/test_data_$3_$4_$1_$2_$i.h5 \
    --method $1 --normalize-weight $2 --input-num-points $3 --num-clusters $4
    echo "data/augmented/test_data_$3_$4_$1_$2_$i.h5" >> data/augmented/test_files_$3_$4_$1_$2.txt
done

