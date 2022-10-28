import csv
import pandas as pd


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        pass 
    return False

def statistic_with_file(file,start,end):
    idx = 0
    data = []
    with open(file,'r',encoding='utf-8') as fp:
        reader=csv.reader(fp)
        for x in reader:
            if idx == 0:
                head = x
            elif idx>=start and idx<=end:
                for i in range(len(x)):
                    if is_number(x[i]):
                        x[i] = float(x[i])
                data.append(x)
            idx+=1
    
    data_df = pd.DataFrame(data, columns = head)
    # print(data_df)
    # TODO: check the parameter in all range is the same!
    
    print("Train F1:", round(data_df["Train"].mean(),4))
    print("Val F1:", round(data_df["Val"].mean(),4))
    print("Test F1:", round(data_df["Test"].mean(),4))

            


if __name__ == "__main__":
    file = "result/ogbg-code2_graphunetpool_[graph_u_net].csv"
    print(file)
    statistic_with_file(file,1,5)