from py2neo import Graph, Node
import math
import tqdm
import time
import xlwt
from itertools import islice
import csv
import random
import numpy as np
import joblib
from os.path import join
import matplotlib.pyplot as plt
from pu_learning import find_stage2_threshold




def test_for_matching():
    graph = Graph('http://localhost:7474/', username='neo4j', password='Myosotis')

    bg_time = time.time()
    print("starting, time: " +str(bg_time))
    match_str = "match(tx:Tx) where tx.BlockID='392512' return tx"
    txs = graph.run(match_str).data()

    print("matching done, time usage: " + str(time.time() - bg_time))

    print(len(txs))
    print(txs[2]['tx'])
    print(type(txs[2]['tx']['NumIn']))
    print(txs)

def feature_statistics():
    feature_path = "./features_differenyears/features_2014.csv"

    feature_paths = ["./features_differenyears/features_2014.csv",
                     "./features_differenyears/features_2015.csv",
                     "./features_differenyears/features_2016.csv"]

    unlabeled_container = []
    positive_container = []

    print("start extracting csvs...")
    for path in feature_paths:
        print(path[-8:-4] + "start...")
        sub_u_container = []
        sub_p_container = []
        csv_reader = islice(csv.reader(open(path)), 1, None)

        for element in tqdm.tqdm(csv_reader):

            element = [float(i) for i in element]
            element = element[1:]

            if element[-1] == 1:
                sub_p_container.append(element)
            else:
                sub_u_container.append(element)

        sub_u_container = np.array(sub_u_container)
        sub_p_container = np.array(sub_p_container)

        unlabeled_container.append(sub_u_container)
        positive_container.append(sub_p_container)



    year = 2014
    for i in range(3):
        print(str(year) + " analysis: ")

        p_input_mean = np.mean(positive_container[i][:, 6])
        p_input_std = np.std(positive_container[i][:, 6])
        p_output_mean = np.mean(positive_container[i][:, 7])
        p_output_std = np.std(positive_container[i][:, 7])
        p_input_money_mean = np.mean(positive_container[i][:, 8])
        p_input_money_std = np.std(positive_container[i][:, 8])
        p_output_money_mean = np.mean(positive_container[i][:, 9])
        p_output_money_std = np.std(positive_container[i][:, 9])
        p_ratio_mean = np.mean(positive_container[i][:, 10])
        p_ratio_std = np.std(positive_container[i][:, 10])

        u_input_mean = np.mean(unlabeled_container[i][:, 6])
        u_input_std = np.std(unlabeled_container[i][:, 6])
        u_output_mean = np.mean(unlabeled_container[i][:, 7])
        u_output_std = np.std(unlabeled_container[i][:, 7])
        u_input_money_mean = np.mean(unlabeled_container[i][:, 8])
        u_input_money_std = np.std(unlabeled_container[i][:, 8])
        u_output_money_mean = np.mean(unlabeled_container[i][:, 9])
        u_output_money_std = np.std(unlabeled_container[i][:, 9])
        u_ratio_mean = np.mean(unlabeled_container[i][:, 10])
        u_ratio_std = np.std(unlabeled_container[i][:, 10])

        print("positive instances statistics: \ninput_number: mean: " + str(p_input_mean) + "  std: " + str(p_input_std) +
              "\noutput_number:  mean: " + str(p_output_mean) + "   std: " + str(p_output_std) +
              "\ninput_money:  mean: " + str(p_input_money_mean) + "   std: " + str(p_input_money_std) +
              "\noutput_money:  mean: " + str(p_output_money_mean) + "   std: " + str(p_output_money_std) +
              "\nratio:  mean: " + str(p_ratio_mean) + "   std: " + str(p_ratio_std))

        print("unlabeled instances statistics: \ninput_number: mean: " + str(u_input_mean) + "  std: " + str(u_input_std) +
            "\noutput_number:  mean: " + str(u_output_mean) + "   std: " + str(u_output_std) +
            "\ninput_money:  mean: " + str(u_input_money_mean) + "   std: " + str(u_input_money_std) +
            "\noutput_money:  mean: " + str(u_output_money_mean) + "   std: " + str(u_output_money_std) +
            "\nratio:  mean: " + str(u_ratio_mean) + "   std: " + str(u_ratio_std))

        print("--------------------------------------------------------------------------------")

    u_total = np.concatenate(unlabeled_container, axis=0)
    p_total = np.concatenate(positive_container, axis=0)
    print("total analysis: ")

    p_input_mean = np.mean(p_total[:, 6])
    p_input_std = np.std(p_total[:, 6])
    p_output_mean = np.mean(p_total[:, 7])
    p_output_std = np.std(p_total[:, 7])
    p_input_money_mean = np.mean(p_total[:, 8])
    p_input_money_std = np.std(p_total[:, 8])
    p_output_money_mean = np.mean(p_total[:, 9])
    p_output_money_std = np.std(p_total[:, 9])
    p_ratio_mean = np.mean(p_total[:, 10])
    p_ratio_std = np.std(p_total[:, 10])

    u_input_mean = np.mean(u_total[:, 6])
    u_input_std = np.std(u_total[:, 6])
    u_output_mean = np.mean(u_total[:, 7])
    u_output_std = np.std(u_total[:, 7])
    u_input_money_mean = np.mean(u_total[:, 8])
    u_input_money_std = np.std(u_total[:, 8])
    u_output_money_mean = np.mean(u_total[:, 9])
    u_output_money_std = np.std(u_total[:, 9])
    u_ratio_mean = np.mean(u_total[:, 10])
    u_ratio_std = np.std(u_total[:, 10])

    print("positive instances statistics: \ninput_number: mean: " + str(p_input_mean) + "  std: " + str(p_input_std) +
              "\noutput_number:  mean: " + str(p_output_mean) + "   std: " + str(p_output_std) +
              "\ninput_money:  mean: " + str(p_input_money_mean) + "   std: " + str(p_input_money_std) +
              "\noutput_money:  mean: " + str(p_output_money_mean) + "   std: " + str(p_output_money_std) +
              "\nratio:  mean: " + str(p_ratio_mean) + "   std: " + str(p_ratio_std))

    print("unlabeled instances statistics: \ninput_number: mean: " + str(u_input_mean) + "  std: " + str(u_input_std) +
            "\noutput_number:  mean: " + str(u_output_mean) + "   std: " + str(u_output_std) +
            "\ninput_money:  mean: " + str(u_input_money_mean) + "   std: " + str(u_input_money_std) +
            "\noutput_money:  mean: " + str(u_output_money_mean) + "   std: " + str(u_output_money_std) +
            "\nratio:  mean: " + str(u_ratio_mean) + "   std: " + str(u_ratio_std))


    print("\n\nAll Done!")

def pu_stage2_demo(model_directory):
    _probability_density = 0.001

    # load data to be evaluated
    val_p_path = "./processed_data/validation/positive.npy"
    val_u_path = "./processed_data/validation/unlabeled.npy"
    train_p_path = "./processed_data/train/raw/train_p.npy"
    train_spy_path = "./processed_data/train/spy.npy"
    train_sub_u_path = "./processed_data/train/sub_u.npy"   # sub unlabeled 0.01 fraction
    train_sub_u_negative_path = "./processed_data/train/sub_u_negative.npy"
    train_u_negative = "./processed_data/train/unlabeled_negative.npy"
    val_p = np.load(val_p_path)[:, :-1]
    val_u = np.load(val_u_path)[:, :-1]
    train_p = np.load(train_p_path)[:, :-1]
    train_spy = np.load(train_spy_path)[:, :-1]
    train_sub_u = np.load(train_sub_u_path)[:, :-1]
    train_sub_u_negative = np.load(train_sub_u_negative_path)[:, :-1]
    train_u_negative = np.load(train_u_negative)[:, :-1]

    classifier = joblib.load(join(model_directory, "logistic.pkl"))

    # classification results
    val_p_result = np.array(classifier.predict_proba(val_p)[:, 1])
    val_u_result = np.array(classifier.predict_proba(val_u)[:, 1])
    train_p_result = np.array(classifier.predict_proba(train_p)[:, 1])
    train_spy_result = np.array(classifier.predict_proba(train_spy)[:, 1])
    train_sub_u_result = np.array(classifier.predict_proba(train_sub_u)[:, 1])
    train_sub_u_negative_result = np.array(classifier.predict_proba(train_sub_u_negative)[:, 1])
    train_u_negative_result = np.array(classifier.predict_proba(train_u_negative)[:, 1])

    # for plots
    val_p_scaled = (val_p_result/_probability_density).astype(np.int16)
    val_u_scaled = (val_u_result/_probability_density).astype(np.int16)
    train_p_scaled = (train_p_result/_probability_density).astype(np.int16)
    train_spy_scaled = (train_spy_result/_probability_density).astype(np.int16)
    train_sub_u_scaled = (train_sub_u_result/_probability_density).astype(np.int16)
    train_sub_u_negative_scaled = (train_sub_u_negative_result/_probability_density).astype(np.int16)
    train_u_negative_scaled = (train_u_negative_result/_probability_density).astype(np.int16)
    val_p_times = [0.0 for i in range(int(1/_probability_density))]
    val_u_times = [0.0 for i in range(int(1/_probability_density))]
    train_p_times = [0.0 for i in range(int(1/_probability_density))]
    train_spy_times = [0.0 for i in range(int(1/_probability_density))]
    train_sub_u_times = [0.0 for i in range(int(1/_probability_density))]
    train_sub_u_negative_times = [0.0 for i in range(int(1/_probability_density))]
    train_u_negative_times = [0.0 for i in range(int(1/_probability_density))]

    for i in tqdm.tqdm(val_p_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        val_p_times[i] += 1
    for i in tqdm.tqdm(val_u_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        val_u_times[i] += 1
    for i in tqdm.tqdm(train_p_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        train_p_times[i] += 1
    for i in tqdm.tqdm(train_spy_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        train_spy_times[i] += 1
    for i in tqdm.tqdm(train_sub_u_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        train_sub_u_times[i] += 1
    for i in tqdm.tqdm(train_sub_u_negative_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        train_sub_u_negative_times[i] += 1
    for i in tqdm.tqdm(train_u_negative_scaled):
        if i >= int(1 / _probability_density):
            i = int(1 / _probability_density) - 1
        train_u_negative_times[i] += 1

    x_axis = np.array([(i+1)*_probability_density for i in range(int(1 / _probability_density))])

    val_p_times = np.array(val_p_times) / val_p.shape[0]
    val_u_times = np.array(val_u_times) / val_u.shape[0]
    train_p_times = np.array(train_p_times) / train_p.shape[0]
    train_spy_times = np.array(train_spy_times) / train_spy.shape[0]
    train_sub_u_times = np.array(train_sub_u_times) / train_sub_u.shape[0]
    train_sub_u_negative_times = np.array(train_sub_u_negative_times) / train_sub_u_negative.shape[0]
    train_u_negative_times = np.array(train_u_negative_times) / train_u_negative.shape[0]

    # plots and save pics
    plt.figure(dpi=400)
    plt.plot(x_axis, val_p_times)
    plt.savefig(join(model_directory, "0514_val_p.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, val_u_times)
    plt.savefig(join(model_directory, "0514_val_u.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, train_p_times)
    plt.savefig(join(model_directory, "0514_train_p.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, train_spy_times)
    plt.savefig(join(model_directory, "0514_train_spy.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, train_sub_u_times)
    plt.savefig(join(model_directory, "0514_train_sub_u.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, train_sub_u_negative_times)
    plt.savefig(join(model_directory, "0514_train_sub_u_negative.png"))

    plt.figure(dpi=400)
    plt.plot(x_axis, train_u_negative_times)
    plt.savefig(join(model_directory, "0514_train_u_negative.png"))



    # evaluating performance

def pu_stage2_calculate_performance(model_directory):
    _negative_threshold = 0.009
    classifier = joblib.load(join(model_directory, "logistic.pkl"))

    print("start finding stage2 threshold...")
    stage2_threshold = find_stage2_threshold(join(model_directory, "logistic.pkl"), 'sub_u')
    print("stage2 threshold found: " + str(stage2_threshold))

    val_p_path = "./processed_data/validation/positive.npy"
    val_u_path = "./processed_data/validation/unlabeled.npy"
    train_p_path = "./processed_data/train/raw/train_p.npy"
    train_sub_u_negative_path = "./processed_data/train/sub_u_negative.npy"
    val_p = np.array(np.load(val_p_path)[:, :-1])
    val_u = np.array(np.load(val_u_path)[:, :-1])
    train_p = np.array(np.load(train_p_path)[:, :-1])
    train_sub_u_negative = np.array(np.load(train_sub_u_negative_path)[:, :-1])

    # filter trustable negative instances
    stageone_classifier = joblib.load("./solver_result/liblinear/0.01/logistic.pkl")
    val_u_stageone_rel = np.array(stageone_classifier.predict_proba(val_u)[:, 1])
    val_u_negative = val_u[np.where(val_u_stageone_rel <= _negative_threshold)]

    val_p_result = np.array(classifier.predict_proba(val_p)[:, 1])
    val_n_result = np.array(classifier.predict_proba(val_u_negative)[:, 1])
    tp_val = np.where(val_p_result >= stage2_threshold, 1, 0).sum() / val_p_result.shape[0]
    fp_val = np.where(val_n_result >= stage2_threshold, 1, 0).sum() / val_n_result.shape[0]
    gmean_val = math.sqrt(tp_val * (1 - fp_val))
    
    train_p_result = np.array(classifier.predict_proba(train_p)[:, 1])
    train_n_result = np.array(classifier.predict_proba(train_sub_u_negative)[:, 1])
    tp_train = np.where(train_p_result >= stage2_threshold, 1, 0).sum() / train_p_result.shape[0]
    fp_train = np.where(train_n_result >= stage2_threshold, 1, 0).sum() / train_n_result.shape[0]
    gmean_train = math.sqrt(tp_train * (1 - fp_train))

    print("batch: " + model_directory)
    print("Train set: TP: " + str(tp_train) + "   FP: " + str(fp_train) + "   GMean: " + str(gmean_train))
    print("Val set: TP: " + str(tp_val) + "   FP: " + str(fp_val) + "   GMean: " + str(gmean_val))






if __name__ == "__main__":
    # pu_stage2_demo("./stage2_result/normal/sag/sub_u")
    pu_stage2_calculate_performance("./stage2_result/normal/lbfgs/sub_u")