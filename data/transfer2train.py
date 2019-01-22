def transfer():
    f1 = open("test_sentiment.txt",'w')
    num = 1
    with open("test_01_03.txt",'r') as f:
        for line in f.readlines():
            datas = line.split("\t")
            if len(datas) == 3:
                sentence = datas[0]
                tags = datas[2]
                f1.writelines(str(num) + "\t" +sentence+"\t"+tags)
                num += 1
            else:
                print(line)
transfer()