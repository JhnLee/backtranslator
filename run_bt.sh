data_dir='./data/imdb_sample.tsv'
output_dir='./output/'
batch_size=64

python main.py \
    --data_dir=${data_dir} \
    --output_dir=${output_dir} \
    --batch_size=${batch_size}