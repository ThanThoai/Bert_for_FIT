from data_util import Preprocess, Dataloader
import argparse
import os
from pytorch_pretrained_bert.modeling import BertForCloth
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch

model_name = "bert-base-uncased"
folder_model = "../../pretrained"
data_dir = "/content/datatest"

parser = argparse.ArgumentParser(description = "Test BERT FIT")
args = parser.parse_args()

args.data_dir = data_dir 
args.pre = args.post = 0
args.bert_model = os.path.join(folder_model, model_name)
args.save_name = f"./data/test-{model_name}.pt"
data = Preprocess(args)


model_test = "PATH_MODEL_TRAINED"
model = BertForCloth.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
model.load_state_dict(torch.load(model_test))
model.to("cuda")
model.eval()

test_data = Dataloader("./data", f"test-{model_name}.pt", 256, 8, "cuda")
test_loss, test_accuracy = 0, 0
nb_test_steps, nb_test_examples = 0, 0

for inp, tgt in test_data.__getitem__():
    with torch.no_grad():
        tmp_test_loss, tmp_test_accuracy = model(inp, tgt)
    test_loss += tmp_test_loss
    test_accuracy += tmp_test_accuracy
    nb_test_examples += inp[-1].sum().item()
    nb_test_steps += 1

test_loss = test_loss / nb_test_steps
test_accuracy = test_accuracy / nb_test_examples

print("LOSS in TEST: ", test_loss)
print("ACCURACY in TEST: ", test_accuracy)