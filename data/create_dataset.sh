wget -O platypus.parquet https://huggingface.co/datasets/garage-bAInd/Open-Platypus/resolve/main/data/train-00000-of-00001-4fe2df04669d1669.parquet?download=true
wget -O sciqa.parquet https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/main/data/train-00000-of-00001-1028f23e353fbe3e.parquet?download=true
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
tar -xvf MATH.tar
rm MATH.tar
wget https://github.com/yuweihao/reclor/releases/download/v1/reclor_data.zip
mkdir reclor_data && unzip -P for_non-commercial_research_purpose_only -d reclor_data reclor_data.zip
git clone https://github.com/mandyyyyii/scibench.git
wget -O oasst1_train.jsonl https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/main/openassistant_best_replies_train.jsonl?download=true
wget -O oasst1_eval.jsonl https://huggingface.co/datasets/timdettmers/openassistant-guanaco/resolve/main/openassistant_best_replies_eval.jsonl?download=true
python get_arb_guanaco.py
python get_leetcode_lima.py
python get_reclor_scibench.py
python get_sciqa_math.py
python get_thm_obqa.py
python merge_jsonls.py
python filter_final_data.py