import torch
#todo
#memory_bank size: (number_patches, patches_size)
#element size: (1, patches_size)
#algo:
#M: memory_bank
#Mc: memory_bank après coreset subsampling
#p: random linear projection (a voir ce qu"on prend)
#pour chaque m E M-Mc:
#calculer min dist(p(m)-p(n)) pour tout n E Mc
#choisir le m correspondant aux maximum des minimums calculés
#virer ce m de M et l'envoyer dans Mc
#faire ça jusqu'à atteindre la taille désirée pour Mc
class Sampler:
    def __init__(self, memory_bank:torch.Tensor, ratio, linear_projection=None):
        self.memory_bank = memory_bank
        self.memory_bank_nrows, self.memory_bank_ncolumns = self.memory_bank.size()
        self.ratio = ratio
        self.linear_projection = linear_projection

    def _compute_subset_lenght(self):
        h, _ = self.memory_bank.size()
        len_subset = int(h * self.ratio)
        self.len_subset = len_subset

    def sample(self):
        """
        in : memory bank, tensor of tensors
        out : subset of memory bank
        """
        mask = torch.zeros(self.memory_bank_nrows, dtype=bool)
        self._compute_subset_lenght()
        for _ in range(self.len_subset):
            try:
                mc = self.memory_bank[mask]
                mem = self.memory_bank[torch.logical_not(mask)]
                d = torch.cdist(mem, mc)
                m = torch.amin(d, dim=1)
                i = torch.argmax(m)
                mask[i] = True

            except:
                #premier coup, pas d'élément dans Mc, on calcule la distance à l'origine
                zeros = torch.zeros((1, self.memory_bank_ncolumns))
                d = torch.cdist(self.memory_bank, zeros)  #size (nrows, 1)
                i = torch.argmax(d, dim=1)
                mask[i] = True
        return self.memory_bank[mask]

