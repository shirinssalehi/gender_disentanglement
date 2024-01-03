import sys
from pandas.io.parquet import read_parquet
# from sentence_transformers import SentenceTransformer, util
import torch
# from sentence_transformers import  LoggingHandler, SentenceTransformer, evaluation, util, models
import pandas as pd
import os
from tqdm import tqdm
# from sentence_transformers import CrossEncoder
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', type=str, required=True)
    parser.add_argument('-queries', type=str, default='', required=True)
    parser.add_argument('-run', type=str, default='', required=True)
    parser.add_argument('-res', type=str, default='', required=True)
    # sys.argv = ["python",
    #              "-checkpoint", "../checkpoints/v1_disentangled_balanced_triples_genderlabelled_passage_label_2e6_gender_dim100_adv_1_lr1_lattr1_ladv0.001_no_disc_adv_sentence-transformers-msmarco-MiniLM-L6-cos-v5-2023-10-23_17-29-29-latest",
    #             "-queries", "/home/shirin/gender_disentanglement/data/215_societal_rekabsaz_dataset.tsv",
    #               "-run", "/home/shirin/gender_disentanglement/runs/run.215_societal_rekabsaz.dev.labelled.trec",
    #                 "-res", "/home/shirin/gender_disentanglement/reranked/testtest.tsv"]
    args = parser.parse_args()

    model_name = args.checkpoint
    model = torch.load(model_name)
    disen = True

    data_folder = "/home/ir-bias/Shirin/msmarco/data/"
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    corpus = {}
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage

    queries_filepath = args.queries
    queries = {}
    with open(queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            queries[qid] = query

    if '.tsv' in args.run:
        run_dev_small = pd.read_csv(args.run, sep = " ", names = ['qid', 'pid', 'rank', 'attribute_label'])
    elif '.trec' in args.run:
        run_dev_small = pd.read_csv(args.run, sep = " ", names = ['qid', 'q0', 'pid', 'rank', 'score', 'retriever', 'attribute_label'])
        run_dev_small.drop(columns=['q0', 'score', 'retriever'], inplace=True)

    
    grps = run_dev_small.groupby('qid')

    reranked_run = []
    scores = []
    pred_attributes = []
    pred_attributes_adv = []
    for name, group in tqdm(grps):
        query = queries[str(name)].strip()
        list_of_docs = []
        for doc_id in group['pid'].values.tolist():
            passage = corpus[str(doc_id)].strip()
            list_of_docs.append((query,passage))
            reranked_run.append([name, doc_id])
        if disen:
            score, pred_attribute, pred_attribute_adv = model.predict(list_of_docs, disen=disen)
        else:
            score, pred_attribute = model.predict(list_of_docs, disen=disen)

        score = score.tolist()
        scores.extend(score)

        if disen:
            pred_attribute = pred_attribute.tolist()
            pred_attributes.extend(pred_attribute)
            pred_attribute_adv = pred_attribute_adv.tolist()
            pred_attributes_adv.extend(pred_attribute_adv)



            

    reranked_run = pd.DataFrame(reranked_run, columns = ['qid', 'pid'])
    reranked_run['score'] = scores
    reranked_run.head()
    reranked_run.sort_values(by=['qid', 'score'], ascending = False, inplace = True)
    reranked_run.to_csv(args.res, sep="\t", index=False, header= None)
    # print(pred_attributes[:100])
    # print(scores[:100])
    # plt.hist(pred_attributes)
    # plt.savefig("../checkpoints/adv_experiments/hist_classification_scores_gender_classifier_3.png")
    # plt.hist(pred_attributes_adv)
    # plt.savefig("../checkpoints/adv_experiments/hist_classification_scores_adv_classifier_3.png")
    if disen:
        pred_attributes_2 = [0 if  pred_attributes[i] <0.5 else 1 for i in range(len(pred_attributes))]
        pred_attributes_adv_2 = [0 if  pred_attributes_adv[i] <0.5 else 1 for i in range(len(pred_attributes_adv))]

        counter = 0
        for i in pred_attributes_2:
            if i ==0:
                counter +=1
                

        true_attributes = run_dev_small["attribute_label"].to_list()

        # Attribute classifier evaluation
        print("Attribute classifier evaluation")
        accuracy = accuracy_score(true_attributes, pred_attributes_2)
        myclassification_report = classification_report(true_attributes, pred_attributes_2)
        conf_matrix = confusion_matrix(true_attributes, pred_attributes_2)

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{myclassification_report}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print("number of zeros={}".format(counter))
        # Adversary Attribute classifier evaluation
        print("-----------------------------------------------------")
        print("Adversary Attribute classifier evaluation")
        accuracy = accuracy_score(true_attributes, pred_attributes_adv_2)
        myclassification_report = classification_report(true_attributes, pred_attributes_adv_2)
        conf_matrix = confusion_matrix(true_attributes, pred_attributes_adv_2)

        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{myclassification_report}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print("number of zeros={}".format(counter))

if __name__ == "__main__":
    main()
    
