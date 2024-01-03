import argparse
import pandas as pd

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run', type=str, default='')
    parser.add_argument('-metric', type=str, default='mrr_cut_10')
#    parser.add_argument('-rrs_path',default='../mrrs/tire_akhar/rrs_only_male_female.csv', type=str)
    args = parser.parse_args()

    metric = args.metric
    k = int(metric.split('_')[-1])
    
    qrel = {}
    with open(args.qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split("\t")
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][did] = int(label)

    run = {}
    with open(args.run, 'r') as f_run:
        for line in f_run:
            qid, did, _ = line.strip().split("\t")
            if qid not in run: 
                run[qid] = []
            run[qid].append(did)
        
    mrr = 0.0
    qids = []
    rrs = []
 #   rrs_df = pd.DataFrame()
    for qid in run:
        rr = 0.0
        for i, did in enumerate(run[qid][:k]):
            if qid in qrel and did in qrel[qid] and qrel[qid][did] > 0:
                rr = 1 / (i+1)
                break
        qids.append(qid)
        rrs.append(rr)
        mrr += rr
    mrr /= len(run)
  #  rrs_df["qid"] = qids
   # rrs_df["rr"] = rrs
    #rrs_df.to_csv(args.rrs_path, index=False)
    print("MRR@10: ", mrr)


if __name__ == "__main__":
    main()
