
class SelfAttention(nn.Module):
    def __init__(self, m, heads=8):
        self.m, self.heads = m, heads

        # We create the key, query and value matrices already stacked
        self.tokeys    = nn.Linear(m, m * heads, bias=False)
        self.toqueries = nn.Linear(m, m * heads, bias=False)
        self.tovalues  = nn.Linear(m, m * heads, bias=False)

        # The final linear transformation to finish newly with m-dimensional vectors
        self.mergeheads = nn.Linear(heads * m, m)

    def forward(self, x):
        b, t, m = x.size()  # batch dimension, sequence length, input vector dimension
        r = self.heads

        # First, we obtain keys, queries, and values
        # we reshape to have a separated dimension for heads
        keys    = self.tokeys(x).view(b, t, r, m)
        queries = self.toqueries(x).view(b, t, r, m)
        values  = self.tovalues(x).view(b, t, r, m)

        # The dot product to obtain the weights should collapse the m dimension
        w_prime = torch.einsum('btrm,bfrm->brtf', queries, keys) / math.sqrt(m)
        w = F.softmax(w_prime, dim=-1)

        # The weighted sum should collapse f-length sequences of m-vectors to single m-vectors (f=t)
        y_conc = torch.einsum('brtf,bfrm->btrm', w, values)

        # Finally we have to merge the outputs from each head, so we should collapse the r dimension (k=m)
        y_conc = torch.einsum('btrm,krm->btk', y_conc, self.mergeheads.weight.view(m,r,m))
        y = y_conc + self.mergeheads.bias
        return y


