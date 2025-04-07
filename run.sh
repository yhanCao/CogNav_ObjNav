for i in {0..27}
do
   # 运行 main.py 并传递 --times 参数
   python3 main.py -d Results/ --skip_times "$i" --scenes '5cdEh9F2hJL'
done
