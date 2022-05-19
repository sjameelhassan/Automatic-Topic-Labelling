import argparse
import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import yaml
import shutil
from operator import itemgetter
import pyLDAvis.gensim_models
from gensim.models import LdaModel, CoherenceModel
from scipy import stats
from extraction import json_extractor
from pathlib import Path
from source.logger import get_logger
logger = get_logger('train_prediction')

class LdaTrainer():
    def __init__(self, input_data, metric):
        #self.source = source
        #self.json_data, self.source_path = retrieve_data(source, make_df=False)
        #self.backup_label = os.path.basename(self.source_path).split('backup-')[1].split('.')[0] # ?
        self.model_id = 'TR.'+str(datetime.now().year)+'-'+str(datetime.now().month) 
        self.metric = metric
        self.input_data = input_data
        self.output_dir = './models/' 
        self.output_filename = '-'.join(self.input_data.split('/')[-1].split('-')[1:])
        self.seed = 1234
        self.chunksize = 2000
        self.passes = 10
       

    def lda_model(self, corpus, num_topics, dictionary):

        model = LdaModel(corpus=corpus, 
                        num_topics=num_topics,  
                        id2word=dictionary,
                        random_state=self.seed,
                        chunksize=self.chunksize,
                        passes=self.passes
                        )
        
        return model
    
    def extract_data(self, model):
        data = pyLDAvis.gensim_models._extract_data( # pylint: disable=protected-access
            model, self.corpus, self.dictionary, doc_topic_dists=None
        )
        return data

    def eval(self, model, metric):
        """ Evaluate LDA model by given number of topic and metric """
        num_topics = len(model.show_topics(num_topics=-1))

        if metric == 'cv':
            cm = CoherenceModel(model = model, texts=self.texts, dictionary=self.dictionary, \
                coherence='c_v', topn=10)
            logger.info('>>> Number of topics: {}'.format(num_topics))
            logger.info('>>> Coherence score: {}'.format(cm.get_coherence()))
            logger.info('>>> Coherence score per topic: {}'.format(cm.get_coherence_per_topic()))
            return cm.get_coherence()

        if metric == 'kl_divergence':
            if self.data is None:
                self.data = self.extract_data(model)
            m1 = self.data['topic_term_dists']
            m2 = self.data['doc_topic_dists']
            L = self.data['doc_lengths']
            C_m1 = np.linalg.norm(m1, axis=1)
            C_m2 = np.matmul(np.expand_dims(L, axis=0), m2)
            C_m1 = C_m1 / np.sum(C_m1)
            C_m2 = C_m2 / np.sum(C_m2)
            C_m1, C_m2 = sorted(np.ndarray.flatten(C_m1)), sorted(np.ndarray.flatten(C_m2))

            logger.info('>>> Number of topics: {}'.format(num_topics))
            kl_divergence = stats.entropy(pk=C_m1, qk=C_m2) + stats.entropy(pk=C_m2, qk=C_m1)
            logger.info('>>> KL divergence: {} \n'.format(kl_divergence))
            
            return kl_divergence

    def train_multiple(self, lower_bound, upper_bound, step):

        topic_counts = range(lower_bound, upper_bound, step)
        transcript_data = [json.loads(line) for line in open(self.input_data, 'r')]
        self.texts, self.corpus, self.dictionary = json_extractor(transcript_data)
        score_all = {}

        for num_topics in topic_counts:
            model = self.lda_model(self.corpus, num_topics, self.dictionary)
            score = self.eval(model, self.metric)
            score_all[num_topics] = score

            folder_path = self.output_dir+'lda_'+str(num_topics)
            #model_path = os.path.join(folder_path, self.model_id +'.lda.model')
            if not Path(folder_path).exists():
                Path(folder_path).mkdir(parents=True, exist_ok=True)

            self.model_path = os.path.join(folder_path, self.model_id +'.lda.model')
            #print(model_path)
            model.save(self.model_path)

            # save model metadata
            metadata_path = os.path.join(folder_path, 'lda.yaml')
            #if not Path(metadata_path).exists():
            metadata = {}
            metadata.update({
                    'model_id': self.model_id,
                    'num_topics': num_topics,
                    'metric': self.metric,
                    'score': float(score),
                    'timestamp': datetime.today()
            })

            with open(metadata_path, 'w') as file:
                  yaml.dump(metadata, file, default_flow_style=False)


            # save topic keywords
            topic_keywords = self.get_topic_keywords(model)
            topic_keywords.to_csv(os.path.join(folder_path, '{}_topic_words.csv'.format(self.model_id)))

        self.best_model(score_all)
        self.prediction(transcript_data)
        return score_all


    def get_topic_keywords(self, model):
        _, topic_words = zip(*model.show_topics(num_topics=-1, num_words=20, formatted=False))
        topic_words = np.asarray(topic_words)[:, :, 0]
        df = pd.DataFrame(topic_words)
        df.index = [self.model_id+'.'+str(i+1) for i in df.index]
        df.columns = ['word{}'.format(i+1) for i in df.columns]
        df.index.name = 'topic_id'
        return df

# Function to automatically identify number of topic based on highest coherence before flattening out

    def best_model(self,score_all):
        max_score = max(score_all.values())
        #max_score = max(score_all)
        
        threshold=0.05

        for key,value in score_all.items():
            score_diff = max_score-value
            if score_diff < threshold:
                best_num_topic= key 
                break
        print(score_all)
        final_model = os.path.join(self.output_dir,"latest")

        if not os.path.exists(final_model):
            os.mkdir(final_model)

        src_folder = os.path.join(self.output_dir,"lda_"+str(best_num_topic))

        files = os.listdir(src_folder)

        for fname in files:
            shutil.copy2(os.path.join(src_folder,fname), final_model)

## predict ( model, input-jsonfile, latest folder,transcript_data)
    def prediction(self, transcript_data):

        prediction_path = './data/prediction'

        if not os.path.exists(prediction_path):
            os.mkdir(prediction_path)

        output_path = os.path.join(prediction_path, 'transcript-'+self.output_filename)

        # load model
        model = LdaModel.load(self.model_path)

        for index,trns_dict in enumerate(transcript_data):
            transcript_tokens = trns_dict['tokens'].split(',')
            transcript_bow = self.dictionary.doc2bow(transcript_tokens)
            topic_prediction = model.get_document_topics(transcript_bow)
            max_prob = sorted(topic_prediction, key=lambda x: x[1], reverse=True)[0]
            transcript_data[index]['topic_id'] = self.model_id+str(max_prob[0])
            transcript_data[index]['probability'] = max_prob[1] 

        with open(output_path, 'w') as tran_json:
            json.dump(str(transcript_data) , tran_json)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--range", nargs=3, type=int, help="range of number of topics (lower bound, upper bound, step)", required=True)
    parser.add_argument("-i", "--input_data", help = "input json data to train model", required = True)
    parser.add_argument("-m", "--metric", help="kl divergence/cv", default="cv")
    args = parser.parse_args()
    logger.info('args: {}'.format(args))


    lda_model = LdaTrainer(
        input_data=args.input_data,
         metric=args.metric       
    )

    score_all = lda_model.train_multiple(lower_bound=args.range[0], upper_bound=args.range[1], step=args.range[2])

    visualization_path = './data/visualization'

    if not os.path.exists(visualization_path):
        os.mkdir(visualization_path)

    score_all_path = os.path.join(visualization_path, 'score_all.json')

    with open(score_all_path, 'w') as score_json:
        json.dump(score_all , score_json)

