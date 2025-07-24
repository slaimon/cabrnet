'''
    Sto cercando di capire per quale motivo il programma non trova nessuna somiglianza tra i campioni e i prototipi,
    nonostante abbia basato i prototipi proprio su quei campioni.

    Questo script replica parti del codice che si occupa della global explaination, per cercare di capire come si
    comporta e quali sono i risultati intermedi.

    Nella global explaination andiamo a confrontare, con ciascun prototipo, il campione dal quale proviene.
    Qui analizziamo soltanto un prototipo: P = 0
'''

import numpy as np
import torch
from cabrnet.core.visualization.upsampling import cubic_upsampling3D
from cabrnet.core.visualization.visualizer3d import SimilarityVisualizer3D
from cabrnet.core.utils.data import DatasetManager
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.save import load_projection_info
from PIL import Image

def main():
    P = 0
    # carico gli oggetti che mi servono: visualizer, dataloader, projection info
    model = CaBRNet.build_from_config("runs/kx_nobackbone_epilogue/final/model_arch.yml")
    visualizer = SimilarityVisualizer3D.build_from_config("configs/protopnet3d/kinetics/explainable/visualization.yml", model)
    dls = DatasetManager.get_dataloaders("configs/protopnet3d/kinetics/explainable/dataset_lite.yml")
    proj = load_projection_info("runs/kx_nobackbone_epilogue/final/projection_info.csv")

    # questo codice si trova in CaBRNet.extract_prototypes.
    # stiamo caricando in img il campione originale, senza augmentazione, normalizzazione o rescaling
    # e in img_tensor il campione con applicate le elaborazioni di cui sopra.
    img = dls["projection_set_raw"].dataset[proj[P]["img_idx"]][0]
    img_tensor = dls["projection_set"].dataset[proj[P]["img_idx"]][0]
    h,w = proj[P]["h"], proj[P]["w"]

    # questa chiamata invece sta in SimilarityVisualizer3D.forward
    sim_map = visualizer.get_attribution(img, img_tensor, P, "cpu", (h,w))

    print(f"shape of similarity map:  {sim_map.shape}")   # (5, 180, 257)
    print(f"max value in sim_map:     {np.max(sim_map)}") # np.float32(0.0)
    print(f"contains nonzero entries? {sim_map.any()}")   # np.False_

    '''
        sim_map ha dimensioni (5, 180, 257) e contiene solo zeri. Siccome la sim_map dovrebbe rappresentare
        "i pixel più simili a un dato prototipo" e sto confrontando in pratica due "immagini" uguali, non
        capisco proprio perché sim_map dovrebbe essere tutta zeri.

        A questo punto vorrei disegnare le due immagini `img` e `img_tensor`.
    '''

    # assumes the tensor has shape [C,T,H,W] and values in range [0,1]
    def to_img(tensor:torch.Tensor) -> Image:
        p = tensor[:,0,:,:]
        p = np.rollaxis(p.numpy()*255, 0,3).astype(np.uint8)
        return Image.fromarray(p)
    
    print("Opening img and img_tensor in default image viewer.")
    to_img(img).show()
    to_img(img_tensor).show()  # le due immagini hanno lo stesso identico aspetto!

    '''
        visualizer.get_attribution restituisce una matrice di soli zeri quando confronta due campioni
        visivamente identici, quindi il problema deve essere lì dentro. Guardiamo il codice dentro il
        metodo get_attribution:
    '''

    attribution_params = visualizer.attribution_params # {'normalize': True, 'location': 'max'}
    attribution_params["location"] = (h,w)

    # A questo punto il metodo chiama la funzione visualizer.attribution == cubic_upsampling3D...
    attr = cubic_upsampling3D(model, img=img, img_tensor=img_tensor, proto_idx=P, device="cpu", **attribution_params)
    # che restituisce la matrice di soli zeri che abbiamo visto prima
    print("Attribution contains nonzero entries?")
    attr.any() # np.False_


    # Perciò dobbiamo guardare dentro cubic_upsampling3D:

    # vogliamo un tensore di forma (1,C,T,H,W) - aggiungiamo la dimensione della batch size così possiamo fare inferenza
    _img_tensor = torch.unsqueeze(img_tensor, dim=0)
    model.eval()

    with torch.no_grad():
        # Compute similarity map (T,H,W)
        sim_map = model.similarities(_img_tensor)[0, P].cpu().numpy()
    
    print("model.similarities returned:")
    print(sim_map) # array([[[0.00233889]]], dtype=float32)
    
    '''
        Il risultato, sim_map, è un tensore di forma 1,1,1!

        Perché stiamo prendendo un tensore 1,1,1 e "normalizzandolo"? è chiaro che il risultato sarà sempre
        [[[0.0]]]. E poi ne facciamo un upscaling alle dimensioni dell'immagine (5,180,257 in questo caso).
        
        Che senso ha?
    '''



# questo serve a non farlo crashare
from multiprocessing import freeze_support
if __name__ == "__main__":
    freeze_support()
    main()