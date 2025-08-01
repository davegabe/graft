# conda create -n GraFT python=3.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.3.0
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt