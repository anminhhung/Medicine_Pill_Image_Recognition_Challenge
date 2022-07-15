pip install -r requirements.txt
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# download pretrained models
gdown --id 1k3bu3rp4aHJA-zQf27XRwRAbvL9LRwXQ
gdown --id 1rnFK3k34XIMeJP6Joe-5qunnBdYualTB

mv craft_mlt_25k.pth saved/models/craft_mlt_25k.pth
mv transformerocr.pth saved/models/transformerocr.pth