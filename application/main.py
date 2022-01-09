from Predictor import Predictor


train_path = 'cale_absoluta/Data/train_samples'
test_path = 'cale_absoluta/Data/test_samples'

a = Predictor(train_path, 10)
a.batch_prediction(test_path)
