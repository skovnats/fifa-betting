import numpy as np
import lightgbm as lgb
import tensorflow as tf

from fifa_ratings_predictor.data_methods import normalise_features

def log_loss(preds, labels):
    """Logarithmic loss with non-necessarily-binary labels."""
    log_likelihood = np.sum(labels * np.log(preds)) / len(preds)
    return -log_likelihood

class NeuralNet:
    def __init__(self, hidden_nodes=8, keep_prob=1.0, learning_rate=0.001):
        self.hidden_nodes = hidden_nodes
        self.keep_prob_value = keep_prob
        self.learning_rate = learning_rate

        self.graph = tf.Graph()
        self.input = None
        self.keep_prob = None
        self.target = None
        self.loss = None
        self.train = None
        self.output = None
        self.training_summary = None
        self.validation_summary = None

        self.build_model()

    def build_model(self):
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32, shape=[None, 39], name='input')
            self.target = tf.placeholder(tf.float32, shape=[None, 3], name='target')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

            hidden_layer = tf.layers.dense(self.input, 16, activation=tf.nn.relu, name="hidden_layer")
            hidden_layer2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, name="hidden_layer2")
            self.output = tf.layers.dense(hidden_layer2, 3, name="output")
            self.output = tf.nn.softmax(self.output, name='softmax')

            with tf.name_scope('losses') as scope:
                self.loss = tf.losses.absolute_difference(self.target, self.output)
                #self.loss = tf.losses.softmax_cross_entropy(self.target, self.output)
                #self.loss = tf.losses.sigmoid_cross_entropy(self.target, self.output)

                self.train = tf.train.MomentumOptimizer(self.learning_rate, 0.99).minimize(self.loss)

            self.training_summary = tf.summary.scalar("training_accuracy", self.loss)
            self.validation_summary = tf.summary.scalar("validation_accuracy", self.loss)

    @staticmethod
    def init_saver(sess):
        writer = tf.summary.FileWriter('./tf-log-SP1/', sess.graph)
        saver = tf.train.Saver(max_to_keep=1)
        return writer, saver

    def train_model(self, X, y, X_val, y_val, model_name):
        best_val_loss = 0.05
        best_val_loss = 100500
        #
        y = np.maximum(y,X[:,-3:])
        y_val = np.maximum(y_val,X_val[:,-3:])
        #y = 
        #X = X[:,:-3]
        #y_val = X_val[:,-3:]
        #X_val = X_val[:,:-3]

        #dtrain = lgb.Dataset(X, y[:,2], free_raw_data=False)
        #dtest = lgb.Dataset(X_val, y_val[:,2], reference=dtrain, free_raw_data=False)
        #evallist = [dtrain, dtest]
        ##
        #train_params = {'learning_rate': 0.05, 'max_depth': 35, 'boosting': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'num_leaves': 10000, 'min_data_in_leaf': 200, 'max_bin': 200, 'colsample_bytree' : 0.7,'subsample': 0.85, 'subsample_freq': 2, 'verbose':-1, 'seed': 1984}
        #gbm = lgb.train(train_params, dtrain, num_boost_round=40000, valid_sets=evallist, early_stopping_rounds=50)

        with tf.Session(graph=self.graph) as sess:

            writer, saver = self.init_saver(sess)

            sess.run(tf.global_variables_initializer())

            for i in range(40000):

                feed_dict = {self.input: X, self.target: y, self.keep_prob: 0.8}

                _, current_loss, train_sum = sess.run([self.train, self.loss, self.training_summary],
                                                      feed_dict=feed_dict)

                if i % 1000 == 0:
                    val_loss, val_sum = sess.run([self.loss, self.validation_summary],
                                                 feed_dict={self.input: X_val, self.target: y_val, self.keep_prob: 1.0})
                    writer.add_summary(val_sum, i)
                    writer.add_summary(train_sum, i)

                    print(i, current_loss, val_loss)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        saver.save(sess, model_name)

    def predict(self, X, model_name):

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_name + '.meta')
            saver.restore(sess, model_name)
            graph = tf.get_default_graph()
            input = graph.get_tensor_by_name('input:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            output = graph.get_tensor_by_name('softmax:0')
            feed_dict = {input: X, keep_prob: 1.0}
            predictions = sess.run(output, feed_dict=feed_dict)

        return predictions

if __name__ == '__main__':

    tf.set_random_seed(8)
    np.random.seed(8)

    league = 'D1'
    #league_1 = 'E0'

    #seasons
    seasons = ['2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018']
    #seasons = ['2013-2014','2014-2015', '2015-2016']

    for i in range(len(seasons)):
        train_seasons = []
        for sn, season in enumerate(seasons):
            if i != sn:
                train_seasons.append(season)

        def get_inp_out(seasons):
            inputs = []
            outputs = []
            for season in seasons:
                inputs.append(np.load(f'./data/lineup-data/{league}/processed-numpy-arrays/feature-vectors-{season}.npy'))
                outputs.append(np.load(f'./data/lineup-data/{league}/processed-numpy-arrays/targets-{season}.npy'))
            inputs = np.vstack(inputs)
            inputs = normalise_features(inputs)

            outputs = np.vstack(outputs).reshape(-1, 3)
            #
            ind = ~np.isnan(inputs).any(axis=1)
            inputs = inputs[ind,:]
            outputs = outputs[ind,:]
            return inputs, outputs



        # inputs = np.vstack((
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2013-2014.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2014-2015.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2015-2016.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2016-2017.npy'),
        #                     ))
        # inputs = normalise_features(inputs)
        # outputs = np.vstack((
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2013-2014.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2014-2015.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2015-2016.npy'),
        #                     np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2016-2017.npy'),
        #                     )).reshape(-1, 3)
        
        print(train_seasons)
        inputs, outputs = get_inp_out(train_seasons)
    #inputs = np.vstack((
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2013-2014.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2014-2015.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2015-2016.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2016-2017.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/feature-vectors-2013-2014.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/feature-vectors-2014-2015.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/feature-vectors-2015-2016.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/feature-vectors-2016-2017.npy'),
    #                    ))
    #inputs = normalise_features(inputs)
    #outputs = np.vstack((
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2013-2014.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2014-2015.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2015-2016.npy'),
    #                    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2016-2017.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/targets-2013-2014.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/targets-2014-2015.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/targets-2015-2016.npy'),
    #                    np.load('./data/lineup-data/' + league_1 + '/processed-numpy-arrays/targets-2016-2017.npy'),
    #                    )).reshape(-1, 3)


    #inputs = np.vstack((
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2013-2014.npy'),
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2014-2015.npy'),
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/feature-vectors-2015-2016.npy')
    #    ))
    #inputs = normalise_features(inputs)
    #outputs = np.vstack((
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2013-2014.npy'),
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2014-2015.npy'),
    #    np.load('./data/lineup-data/' + league + '/processed-numpy-arrays/targets-2015-2016.npy')
    #    )).reshape(-1, 3)


        nan_rows = np.where(outputs != outputs)[0]

        inputs = np.delete(inputs, nan_rows, axis=0)
        outputs = np.delete(outputs, nan_rows, axis=0)

        #outputs = 1 / outputs

        net = NeuralNet()

        net.train_model(inputs[:-50], outputs[:-50], inputs[-50:], outputs[-50:], model_name=f'./models/{league}-{seasons[i]}' +
                                                                                    '/deep')

    # net = NeuralNet()

    #predictions = net.predict(inputs[-50:], model_name='./models/' + league + '/deep')

    #for i, j in zip(predictions, outputs[-50:]):
    #    print(i)
    #    print(j)
