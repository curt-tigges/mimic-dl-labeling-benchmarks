import numpy as np
import pickle

############################################
# Pickle save & load test
############################################
TEST_PICKLE_PATH = './pickle_test.pkl'

arr = np.array([[0.00500028], [0.3401312]], dtype=np.float32)
with open(TEST_PICKLE_PATH, 'wb') as f:  # Should be binary mode!! w"b"
    pickle.dump(arr, f)

with open(TEST_PICKLE_PATH, 'rb') as f:    # Should be binary mode!! r"b"
    org_arr = pickle.load(f)
    print(org_arr)

############################################
# Multilabel Classification ROC Curve
############################################
