cd ./data/
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
cd ../models
wget https://www.dropbox.com/s/5w7skbqq1hr2xwu/model_epoch_90.ckpt.data-00000-of-00001
wget https://www.dropbox.com/s/5th42p5us3o7gy2/model_epoch_90.ckpt.index
wget https://www.dropbox.com/s/9ifjkdkl7l3q74l/model_epoch_90.ckpt.meta
cd ../
python3 generate.py --model=./models/model_epoch_90.ckpt --caption_file=$1
