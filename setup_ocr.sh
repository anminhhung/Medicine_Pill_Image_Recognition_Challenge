pip install -r requirements.txt

# download pretrained models
gdown --id 1k3bu3rp4aHJA-zQf27XRwRAbvL9LRwXQ
gdown --id 1rnFK3k34XIMeJP6Joe-5qunnBdYualTB

mv craft_mlt_25k.pth saved/models/craft_mlt_25k.pth
mv transformerocr.pth saved/models/transformerocr.pth