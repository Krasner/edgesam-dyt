import torch

sam_weight = torch.load('../weights/sam_vit_h_4b8939.pth')

for key_word in ['mask_decoder','prompt_encoder']:
    new_weight = {}
    for key in sam_weight.keys():
        if key_word in key:
            new_weight[key] = sam_weight[key]

    torch.save(new_weight, f'../weights/sam_vit_h_{key_word}.pth')