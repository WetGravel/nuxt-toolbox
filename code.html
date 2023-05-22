class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):

        super(GDN, self).__init__()

        node_names=[]
        with open('./data/msl/list.txt') as f:
            node_names=f.read().splitlines()




        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        

        self.node_embedding = None
        self.topk = 0


        
        adj_csv=readCSV2List("./data/msl/dl_am.csv")[1:-1]#读邻接矩阵
        csv_node_names=readCSV2List("./data/msl/dl_am.csv")[0]#读取节点名
        adj_matrix=[]
        
        # for i in range(len(node_names)):
        #     if node_names[i] not in csv_node_names:
        #         continue

        adj_index=-1
        for i in range(len(adj_csv)-1):#查找邻接矩阵生成list
            if adj_csv[i][0] not in node_names:#查找邻接矩阵csv第一列在list.txt中的话开始补充有邻接关系的值
                continue
            adj_matrix.append([])
            adj_index=adj_index+1
            for j in range(len(adj_csv[0])):
                if j>0 and adj_csv[i][j]=='1' and csv_node_names[j] in node_names:
                    adj_matrix[adj_index].append(node_names.index(csv_node_names[j]))

        for i in range(len(adj_matrix)):#计算最长的地方
            if(self.topk<len(adj_matrix[i])):
                self.topk=len(adj_matrix[i])
                              
                              
        for i in range(len(adj_matrix)):#每个list补成topk个
           for j in range(self.topk-len(adj_matrix[i])):
               adj_matrix[i].append(i)



        self.adi_torch=torch.tensor(adj_matrix,dtype=torch.int64).to(device)        

        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        out_module=[]
        
        out_module.append(nn.Linear(30,2))
        out_module.append(nn.Softmax(-1))
        self.res_mod=nn.ModuleList(out_module)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=1.0)


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            # topk_indices_ji=self.adi_torch
            # for k in range(len(self.adi_torch)):#将空缺的地方用计算得到的相关性高的没出现过的节点补上
            #     for j in range(len(self.adi_torch[0])):
            #         if(self.adi_torch[k][j]==-1):
            #             for p in range(len(topk_indices_ji[k])):
            #                 if topk_indices_ji[k][p] not in self.adi_torch[k]:
            #                      self.adi_torch[k][j]=topk_indices_ji[k][j]

            topk_indices_ji=self.adi_torch

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

           
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            
            #print(x.size())
            #print(batch_edge_index.size())
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        
        #out=torch.squeeze(data,dim=-1)
        #out=torch.cat([out,data],dim=1)
        #out=torch.flatten(out,1)
        for mod in self.res_mod:
              out=mod(out)
        
        return out
