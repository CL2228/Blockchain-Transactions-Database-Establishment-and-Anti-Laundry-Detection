import numpy as np
from sklearn.linear_model._logistic import LogisticRegression, LogisticRegressionCV
import csv
from itertools import islice
import tqdm
import joblib
from os.path import join
import matplotlib.pyplot as plt
import math

_spy_rate = 0.15
_sub_u_rates = [0.01, 0.05]      # only use a sub set of unlabeled data for training, randomly chosen
train_p_path = "./processed_data/train/raw/train_positive_set.csv"
train_u_path = "./processed_data/train/raw/train_unlabeled_set.csv"
_spy_path = "./processed_data/train/spy.npy"
_sub_u_path = "./processed_data/train/sub_u.npy"
_unlabeled_path = "./processed_data/train/raw/train_u.npy"
_p_path = "./processed_data/train/raw/train_p.npy"
_solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag']
_result_path = './solver_result'
_stage_one_gradient_step = 0.001
_negative_threshold = 0.009
_threshold_2_step = 0.0001
_stage2_result_path = "./stage2_result/balanced/1e-3"
_bins_num = 300     # for ploting
_max_iteration = 100        # for LR in stage 2
_l2_coefficient = 0.001


def spy_train_dataset():
    """
    actually find it useless
    :return:
    """
    train_p = np.load("./processed_data/train/raw/train_p.npy")
    train_u = np.load("./processed_data/train/raw/train_u.npy")
    print(train_p.shape)
    print(train_u.shape)

    np.random.shuffle(train_p)
    spy = train_p[: int(_spy_rate * train_p.shape[0]), :]
    spy[:, -1] = 0

    spy_u = np.concatenate([train_u, spy])
    spy_p = np.copy(train_p[int(_spy_rate * train_p.shape[0]):])
    print(spy_p.shape)
    print(spy_u.shape)
    spy_train = np.concatenate([spy_p, spy_u])
    np.random.shuffle(spy_train)
    np.save("./processed_data/train/spy/train.npy", spy_train)
    print(spy_train.shape)


def csv2npy():
    """
    transform data from csv to npy
    :return:
    """
    train_p_reader = csv.reader(open(train_p_path))
    train_u_reader = csv.reader(open(train_u_path))

    p_list = []
    u_list = []

    for ele in tqdm.tqdm(islice(train_p_reader, 1, None)):
        p_list.append([float(i) for i in ele[1:]])
    for ele in tqdm.tqdm(islice(train_u_reader, 1, None)):
        u_list.append([float(i) for i in ele[1:]])

    p_npy = np.array(p_list)
    u_npy = np.array(u_list)
    np.save("./processed_data/train/raw/train_p.npy", p_npy)
    np.save("./processed_data/train/raw/train_u.npy", u_npy)
    print(u_npy[-50:])


def creat_npy():
    """
    create npy of each sub-unlabeled set
    :return:
    """
    train_p = np.load("./processed_data/train/raw/train_p.npy")
    train_u = np.load("./processed_data/train/raw/train_u.npy")
    np.random.shuffle(train_u)  # shuffle
    np.random.shuffle(train_p)

    # spy instances, replace the label of spies, now spies are marked as 0
    spy = train_p[: int(_spy_rate * train_p.shape[0]), :]
    spy[:, -1] = 0
    np.save(_spy_path, spy)  # store spy for later evaluation

    # only use a subset of unlabeled set for training (time saving purpose)
    sub_u_1 = train_u[: int(0.01 * train_u.shape[0])]
    np.save("./processed_data/train/sub_u_0.01.npy", sub_u_1)

    sub_u_2 = train_u[: int(0.05 * train_u.shape[0])]
    np.save("./processed_data/train/sub_u_0.05.npy", sub_u_2)

    sub_u_3 = train_u[: int(0.1 * train_u.shape[0])]
    np.save("./processed_data/train/sub_u_0.1.npy", sub_u_3)

    sub_u_4 = train_u[: int(0.5 * train_u.shape[0])]
    np.save("./processed_data/train/sub_u_0.5.npy", sub_u_4)


def pu_first_stage_training(solver, sub_u_rate):
    """
    first stage of PU-Learning, including LR training of stage one, data distributing, model saving
    :param solver:
    :param sub_u_rate:
    :return:
    """
    train_p = np.load("./processed_data/train/raw/train_p.npy")
    train_u = np.load("./processed_data/train/raw/train_u.npy")
    np.random.shuffle(train_u)      # shuffle
    np.random.shuffle(train_p)

    # spy instances, replace the label of spies, now spies are marked as 0
    spy = train_p[: int(_spy_rate * train_p.shape[0]), :]
    spy[:, -1] = 0
    np.save(_spy_path, spy)     #store spy for later evaluation

    # only use a subset of unlabeled set for training (time saving purpose)
    sub_u = train_u[: int(sub_u_rate * train_u.shape[0])]
    np.save(_sub_u_path, sub_u)
    np.save("./processed_data/train/sub_u_" +str(sub_u_rate) + ".npy", sub_u)

    train_spy_u = np.concatenate([spy, sub_u])      # a set that contains spies and sub-unlabeled
    train = np.concatenate([train_p[int(_spy_rate * train_p.shape[0]):], train_spy_u])  # the whole training set

    print(train.shape)

    # shuffle the training set
    np.random.shuffle(train)
    train_X = train[:, :-1]
    label_X = train[:, -1]

    # logistic regression, using sag optimization method
    classifier = LogisticRegression(solver=solver)
    classifier.fit(train_X, label_X)

    # total = train_X.shape[0]
    # true = 0
    # false = 0
    # a rough evaluation
    # for i in tqdm.tqdm(range(train_X.shape[0])):
    #     predict = classifier.predict_proba(np.array([train_X[i]]))
    #
    #     if abs(predict[0][1] - label_X[i]) < 0.5:
    #         true += 1
    #     else:
    #         false += 1
    # print(true/total)

    # save the model
    joblib.dump(classifier, join(join(_result_path, solver), str(sub_u_rate)) + '/logistic.pkl')


def first_stage_test(solver, sub_u_rate):
    """
    evaluation function of stage one, aiming to visualize the probability distribution of LR in each set
    :param solver:
    :param sub_u_rate:
    :return:
    """
    bins_num = 100

    classifier = joblib.load(join(join(_result_path, solver), str(sub_u_rate)) + '/logistic.pkl')

    # evaluate positive set, which contains spies
    positive = np.load("./processed_data/train/raw/train_p.npy")
    positive_x = positive[:, : -1]
    result_p = np.array(classifier.predict_proba(positive_x)[:, 1])
    plt.hist(result_p, bins=bins_num)
    plt.savefig(join(join(_result_path, solver), str(sub_u_rate)) + '/positive.png')
    plt.show()
    print("\npositive set results: average: " + str(np.mean(result_p)) + "   variance:" + str(np.var(result_p)))
    print("max: " + str(result_p.max()) + "  min: " + str(result_p.min()))

    # evaluate spy set
    spy = np.load(_spy_path)
    spy_x = spy[:, : -1]
    result_spy = np.array(classifier.predict_proba(spy_x)[:, 1])
    plt.hist(result_spy, bins=bins_num)
    plt.savefig(join(join(_result_path, solver), str(sub_u_rate)) + '/spy.png')
    plt.show()
    print("\nspy set results: average: " + str(np.mean(result_spy)) + "   variance:" + str(np.var(result_spy)))
    print("max: " + str(result_spy.max()) + "  min: " + str(result_spy.min()))

    # evaluate sub-unlabeled set
    sub_u = np.load("./processed_data/train/sub_u_" + str(sub_u_rate) + ".npy")
    sub_u_x = sub_u[:, :-1]
    result_sub_u = np.array(classifier.predict_proba(sub_u_x)[:, 1])
    plt.hist(result_sub_u, bins=bins_num)
    plt.savefig(join(join(_result_path, solver), str(sub_u_rate)) + '/sub-u.png')
    plt.show()
    print("\nsub-unlabeled set results: average: " + str(np.mean(result_sub_u)) + "   variance:" + str(np.var(result_sub_u)))
    print("max: " + str(result_sub_u.max()) + "  min: " + str(result_sub_u.min()))

    # evaluate the whole unlabeled set
    unlabeled = np.load("./processed_data/train/raw/train_u.npy")
    unlabeled_x = unlabeled[:, :-1]
    result_unlabeled = np.array(classifier.predict_proba(unlabeled_x)[:, 1])
    plt.hist(result_unlabeled, bins=bins_num)
    plt.savefig(join(join(_result_path, solver), str(sub_u_rate)) + '/unlabeled.png')
    plt.show()
    print("\nunlabeled set results: average: " + str(np.mean(result_unlabeled)) + "   variance:" + str(
        np.var(result_unlabeled)))
    print("max: " + str(result_unlabeled.max()) + "  min: " + str(result_unlabeled.min()))


def train_and_eva():
    """
    just an iteration of training and evaluation
    :return:
    """
    for sol in _solvers:
        for sub_u_rate in _sub_u_rates:
            print("now processing " + sol + "  " + str(sub_u_rate))
            pu_first_stage_training(sol, sub_u_rate)
            first_stage_test(sol, sub_u_rate)
            print("\n\n")


def find_negative_threshold(solver, sub_u_rate):
    """
    find the threshold of negative instances in stage one using the deference of CDF
    :param solver:
    :param sub_u_rate:
    :return:
    """
    classifier = joblib.load(join(join(_result_path, solver), str(sub_u_rate)) + '/logistic.pkl')

    spy_times = [0.0 for i in range(int(1/_stage_one_gradient_step))]
    sub_u_times = [0.0 for i in range(int(1/_stage_one_gradient_step))]

    spy = np.load(_spy_path)
    spy_x = spy[:, : -1]
    result_spy = np.array(classifier.predict_proba(spy_x)[:, 1])
    result_spy = result_spy / _stage_one_gradient_step
    result_spy = result_spy.astype(np.int16)

    sub_u = np.load("./processed_data/train/sub_u_" + str(sub_u_rate) + ".npy")
    sub_u_x = sub_u[:, :-1]
    result_sub_u = np.array(classifier.predict_proba(sub_u_x)[:, 1])
    result_sub_u = result_sub_u / _stage_one_gradient_step
    result_sub_u = result_sub_u.astype(np.int16)

    for i in tqdm.tqdm(result_spy):
        spy_times[i] += 1
    for i in tqdm.tqdm(result_sub_u):
        sub_u_times[i] += 1

    spy_times = np.array(spy_times) / spy.shape[0]
    sub_u_times = np.array(sub_u_times) / sub_u.shape[0]

    x_axis = np.array([(i+1)*_stage_one_gradient_step for i in range(int(1 / _stage_one_gradient_step))])

    cumulative_spy = np.cumsum(spy_times)
    cumulative_sub_u = np.cumsum(sub_u_times)

    gradient_spy = [0 for i in range(int(1 / _stage_one_gradient_step))]
    gradient_sub_u = [0 for i in range(int(1 / _stage_one_gradient_step))]

    for i in range(1, cumulative_spy.shape[0]):
        gradient_spy[i] = cumulative_spy[i] - cumulative_spy[i - 1]
        gradient_sub_u[i] = cumulative_sub_u[i] - cumulative_sub_u[i - 1]


    gradient_spy = np.array(gradient_spy)
    gradient_sub_u = np.array(gradient_sub_u)

    gradient_minus = gradient_sub_u - gradient_spy


    # plt.plot(x_axis, spy_times)
    # plt.show()
    # plt.plot(x_axis, sub_u_times)
    # plt.show()
    # plt.plot(x_axis, cumulative_spy)
    # plt.show()
    # plt.plot(x_axis, cumulative_sub_u)
    # plt.show()
    # plt.plot(x_axis, gradient_spy)
    # plt.show()
    # plt.plot(x_axis, gradient_sub_u)
    # plt.show()
    # plt.plot(x_axis, gradient_minus)
    # plt.show()
    threshold = gradient_minus.argmax() * _stage_one_gradient_step
    print(threshold)


def filter_negative(solver, sub_u_rate, threshold):
    """
    filter negative instances from sub-unlabeled set and unlabeled set
    :param solver:
    :param sub_u_rate:
    :param threshold:
    :return:
    """
    classifier = joblib.load(join(join(_result_path, solver), str(sub_u_rate)) + '/logistic.pkl')

    sub_u = np.load("./processed_data/train/sub_u_" + str(sub_u_rate) + ".npy")
    sub_u_x = sub_u[:, :-1]
    result_sub_u = np.array(classifier.predict_proba(sub_u_x)[:, 1])

    sub_u_negative = sub_u[np.where(result_sub_u <= threshold)]
    print(sub_u_negative.shape)
    sub_u_negative_x = sub_u_negative[:, :-1]
    result_sub_u_negative = np.array(classifier.predict_proba(sub_u_negative_x)[:, 1])
    print(result_sub_u_negative.max())
    np.save("./processed_data/train/sub_u_negative.npy", sub_u_negative)

    print("\n\n\n")
    unlabeled = np.load("./processed_data/train/raw/train_u.npy")
    unlabeled_x = unlabeled[:, :-1]
    result_unlabeled = np.array(classifier.predict_proba(unlabeled_x)[:, 1])
    unlabeled_negative = unlabeled[np.where(result_unlabeled <= threshold)]
    print(unlabeled_negative.shape)
    result_unlabeled_negative = np.array(classifier.predict_proba(unlabeled_negative[:, :-1])[:, 1])
    print(result_unlabeled_negative.max())
    np.save("./processed_data/train/unlabeled_negative.npy", unlabeled_negative)


def stage_2_training(solver):
    """
    training LR process in stage two using positive and negative instances, with Logistic Regression
    :param solver:
    :return:
    """
    positive = np.load("./processed_data/train/raw/train_p.npy")
    sub_u_negative = np.load("./processed_data/train/sub_u_negative.npy")
    unlabeled_negative = np.load("./processed_data/train/unlabeled_negative.npy")

    # only use the sub-u set for training
    train_p_subu = np.concatenate([positive, sub_u_negative])
    np.random.shuffle(train_p_subu)
    x_train_p_subu = train_p_subu[:, :-1]
    y_train_p_subu = train_p_subu[:, -1]
    classifier = LogisticRegression(solver=solver,
                                    class_weight='balanced', penalty='l2', max_iter=_max_iteration, C=_l2_coefficient)
    classifier.fit(x_train_p_subu, y_train_p_subu)

    image_dir = _stage2_result_path + "/" + solver + "/sub_u/"
    result_p = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
    plt.hist(result_p, bins=_bins_num)
    plt.savefig(image_dir + "train_positive.png")
    plt.show()
    result_sub_u = np.array(classifier.predict_proba(sub_u_negative[:, :-1])[:,1])
    plt.hist(result_sub_u, bins=_bins_num)
    plt.savefig(image_dir + "train_sub_u.png")
    plt.show()
    model_path = _stage2_result_path + "/" + solver + "/sub_u/logistic.pkl"
    joblib.dump(classifier, model_path)

    # use negative instances from the whole unlabeled set for training
    train_p_unlabeled = np.concatenate([positive, unlabeled_negative])
    np.random.shuffle(train_p_unlabeled)
    x_train_p_unlabeled = train_p_unlabeled[:, :-1]
    y_train_p_unlabeled = train_p_unlabeled[:, -1]
    classifier = LogisticRegression(solver=solver,
                                    class_weight='balanced', penalty='l2', max_iter=_max_iteration, C=_l2_coefficient)
    classifier.fit(x_train_p_unlabeled, y_train_p_unlabeled)
    result_p = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
    image_dir = _stage2_result_path + "/" + solver + "/unlabeled/"
    plt.hist(result_p, bins=_bins_num)
    plt.savefig(image_dir + "train_positive.png")
    plt.show()
    result_unlabeled = np.array(classifier.predict_proba(unlabeled_negative[:, :-1])[:,1])
    plt.hist(result_unlabeled, _bins_num)
    plt.savefig(image_dir + "train_unlabeled.png")
    plt.show()
    model_path = _stage2_result_path + "/" + solver + "/unlabeled/logistic.pkl"
    joblib.dump(classifier, model_path)


def find_stage2_threshold(model_path, u_mode):
    """
        determine the threshold of stage 2 by iteration,
    :param model_path:  path of LR model
    :param u_mode:  'sub_u' or 'unlabeled'
    :return:
    """
    u_modes = ['sub_u', 'unlabeled']

    assert u_mode in u_modes

    if u_mode == 'sub_u':
        negative = np.load("./processed_data/train/sub_u_negative.npy")
    else:
        negative = np.load("./processed_data/train/unlabeled_negative.npy")

    positive = np.load("./processed_data/train/raw/train_p.npy")


    # begin with sub-u classifier test
    classifier = joblib.load(model_path)
    u_bst_th = 0.0
    u_th = 0.0
    u_index = 0.0
    u_tp = 0.0
    u_fp = 0.0
    while u_th < 1:
        p_result = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
        sub_u_result = np.array(classifier.predict_proba(negative[:, :-1])[:, 1])
        tp_rate = np.where(p_result >= u_th, 1, 0).sum() / p_result.shape[0]
        fp_rate = np.where(sub_u_result >= u_th, 1, 0).sum() / sub_u_result.shape[0]
        index = math.sqrt(tp_rate * (1 - fp_rate))
        if index >= u_index:
            u_index = index
            u_tp = tp_rate
            u_fp = fp_rate
            u_bst_th = u_th
        print("threshold: " + str(u_th) + "   TP: "
              + str(tp_rate) + "   FP: " + str(fp_rate) + "   GMean: " + str(index) + "\n\n")
        u_th += _threshold_2_step



    print(model_path +  "\n " +
          "threshold: " + str(u_bst_th) + "   TP: " + str(u_tp) + "   FP: " + str(u_fp) + "   GMean: " + str(u_index))



def evaluation(model_path, threshold):
    """
    final evaluation process of PU-Learning
    :param model_path:
    :param threshold:
    :return:
    """
    classifier = joblib.load(model_path)

    positive = np.load("./processed_data/validation/positive.npy")
    unlabeled = np.load("./processed_data/validation/unlabeled.npy")

    p_result = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
    plt.hist(p_result, bins=300)
    plt.show()

    tp_rate = np.where(p_result >= threshold, 1, 0).sum() / p_result.shape[0]
    print(tp_rate)

    u_result = np.array(classifier.predict_proba(unlabeled[:, :-1])[:, 1])
    plt.hist(u_result, bins=300)
    plt.show()


    # the following steps aim to filter 'possible' negative instances in the evaluation-unlabeled set
    stageone_classifier = joblib.load("./solver_result/liblinear/0.01/logistic.pkl")
    stgone_result = np.array(stageone_classifier.predict_proba(unlabeled[:,:-1])[:, 1])
    possibly_negative = unlabeled[np.where(stgone_result <= _negative_threshold)]
    print(positive.shape)
    print(unlabeled.shape)
    print(possibly_negative.shape)
    possi_ng_result = np.array(classifier.predict_proba(possibly_negative[:, :-1])[:, 1])
    fp_rate = np.where(possi_ng_result >= threshold, 1, 0).sum() / possi_ng_result.shape[0]
    plt.hist(possi_ng_result, bins=300)
    plt.show()

    print(fp_rate)
    print("TP: " + str(tp_rate) + "   FP: " + str(fp_rate) + "   GMean: " + str(math.sqrt(tp_rate * (1 - fp_rate))))

def pdf_of_validation(solver):
    """
    this function aims to process the probability distribution function image of different models
    sub_u model not needed, automatically process both sub_u and unlabeled
    :param solver:  the optimization method of LR
    :return:  nothing returned, but images saved
    """

    # load evaluation sets
    positive = np.load("./processed_data/validation/positive.npy")
    unlabeled = np.load("./processed_data/validation/unlabeled.npy")

    # first process sub_u mode
    model_path = _stage2_result_path + "/" + solver + "/sub_u/logistic.pkl"
    classifier = joblib.load(model_path)
    image_path = _stage2_result_path + "/" + solver + "/sub_u/"
    p_result = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
    plt.hist(p_result, bins=_bins_num)
    plt.savefig(image_path + "val_positive.png")
    plt.show()
    u_result = np.array(classifier.predict_proba(unlabeled[:, :-1])[:, 1])
    plt.hist(u_result, bins=_bins_num)
    plt.savefig(image_path + "val_unlabeled.png")
    plt.show()


    # next process unlabeled mode
    model_path = _stage2_result_path + "/" + solver + "/unlabeled/logistic.pkl"
    classifier = joblib.load(model_path)
    image_path = _stage2_result_path + "/" + solver + "/unlabeled/"
    p_result = np.array(classifier.predict_proba(positive[:, :-1])[:, 1])
    plt.hist(p_result, bins=_bins_num)
    plt.savefig(image_path + "val_positive.png")
    plt.show()
    u_result = np.array(classifier.predict_proba(unlabeled[:, :-1])[:, 1])
    plt.hist(u_result, bins=_bins_num)
    plt.savefig(image_path + "val_unlabeled.png")
    plt.show()


def stage2_iterator():
    for solver in ['liblinear', 'lbfgs']:
        print(solver + " training processing begins...")
        stage_2_training(solver)
        print(solver + " training done, now start evaluating...")
        pdf_of_validation(solver)
        print(solver + " evaluation done!\n\n")



if __name__ == "__main__":
    # find_stage2_threshold("./stage2_result/balanced/1e-3/liblinear/sub_u/logistic.pkl", "sub_u")
    evaluation("./stage2_result/balanced/1e-3/liblinear/sub_u/logistic.pkl", 0.5648)
    # find_negative_threshold("liblinear", 0.01)