import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, BaseModelOutputWithPastAndCrossAttentions
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from einops import rearrange
from utils.RevIN import RevIN
from utils.embed import DataEmbedding_wo_time
from transformers.models.gpt2.modeling_gpt2 import *
from typing import Optional, Tuple, Union
import math


class RelationalGCNLayer(nn.Module):
    """Relational GCN layer for multiple edge types"""
    def __init__(self, in_channels, out_channels, num_relations, activation=True):
        super(RelationalGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.activation = activation
        
        # Weight matrices for each relation type
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # Self-loop weight
        self.self_loop_weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop_weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_type, edge_weight=None):
        N = x.size(0)
        out = torch.matmul(x, self.self_loop_weight)

        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue
            r_edge_index = edge_index[:, mask]
            r_edge_weight = edge_weight[mask] if edge_weight is not None else None

            x_transformed = torch.matmul(x, self.weight[r])

            row, col = r_edge_index

            # Degree normalization: D^{-1/2} A D^{-1/2} per relation
            deg = torch.zeros(N, device=x.device)
            deg.index_add_(0, row, torch.ones(mask.sum(), device=x.device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg == 0] = 0.0

            out_r = torch.zeros_like(out)
            msgs = x_transformed[col] * deg_inv_sqrt[col].unsqueeze(-1)
            if r_edge_weight is not None:
                msgs = msgs * r_edge_weight.unsqueeze(-1)
            out_r.index_add_(0, row, msgs)
            out += out_r * deg_inv_sqrt.unsqueeze(-1)

        out += self.bias

        if self.activation:
            out = F.relu(out)

        return out


class HPRG(nn.Module):
    """Heterogeneous Patch Relation Graph"""
    def __init__(self, d_model, gnn_hidden, gnn_layers, k_sim=5, 
                 gate_init=0.5, num_relations=4, dropout=0.1):
        super(HPRG, self).__init__()
        
        self.d_model = d_model
        self.gnn_hidden = gnn_hidden
        self.gnn_layers = gnn_layers
        self.k_sim = k_sim
        self.num_relations = num_relations
        
        self.gnn_layers_list = nn.ModuleList()
        
        for i in range(gnn_layers):
            if i == 0:
                in_dim = self.d_model
            else:
                in_dim = self.gnn_hidden
                
            if i == gnn_layers - 1:
                out_dim = self.d_model
            else:
                out_dim = self.gnn_hidden
            
            layer = RelationalGCNLayer(in_dim, out_dim, num_relations)
            self.gnn_layers_list.append(layer)
        
        self.layer_norm = nn.LayerNorm(d_model)
        # Learnable gate for residual connection
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self.dropout = nn.Dropout(dropout)
        
    def build_patch_graph(self, batch_size, num_vars, num_patches, device):
        """Build the patch relation graph using vectorized operations"""
        edge_list = []
        edge_type_list = []
        
        # Create node indices tensor
        b_idx = torch.arange(batch_size, device=device)
        v_idx = torch.arange(num_vars, device=device)
        p_idx = torch.arange(num_patches, device=device)
        
        # Type 0: Temporal adjacency within same variable
        if num_patches > 1:
            b_expand = b_idx.view(-1, 1, 1).expand(batch_size, num_vars, num_patches-1)
            v_expand = v_idx.view(1, -1, 1).expand(batch_size, num_vars, num_patches-1)
            p_expand = p_idx[:-1].view(1, 1, -1).expand(batch_size, num_vars, num_patches-1)
            
            src_nodes = b_expand * (num_vars * num_patches) + v_expand * num_patches + p_expand
            dst_nodes = src_nodes + 1
            
            src_nodes_flat = src_nodes.flatten()
            dst_nodes_flat = dst_nodes.flatten()
            
            edge_list.append(torch.stack([src_nodes_flat, dst_nodes_flat], dim=0))
            edge_type_list.append(torch.zeros(src_nodes_flat.shape[0], dtype=torch.long, device=device))
            edge_list.append(torch.stack([dst_nodes_flat, src_nodes_flat], dim=0))
            edge_type_list.append(torch.zeros(dst_nodes_flat.shape[0], dtype=torch.long, device=device))
        
        # Type 1: Cross-variable synchrony at same time
        if num_vars > 1:
            v1_mesh, v2_mesh = torch.meshgrid(v_idx, v_idx, indexing='ij')
            cross_var_mask = v1_mesh != v2_mesh
            v1_pairs = v1_mesh[cross_var_mask]
            v2_pairs = v2_mesh[cross_var_mask]
            
            num_pairs = v1_pairs.shape[0]
            b_expand = b_idx.view(-1, 1, 1).expand(batch_size, num_pairs, num_patches)
            v1_expand = v1_pairs.view(1, -1, 1).expand(batch_size, num_pairs, num_patches)
            v2_expand = v2_pairs.view(1, -1, 1).expand(batch_size, num_pairs, num_patches)
            p_expand = p_idx.view(1, 1, -1).expand(batch_size, num_pairs, num_patches)
            
            src_nodes = b_expand * (num_vars * num_patches) + v1_expand * num_patches + p_expand
            dst_nodes = b_expand * (num_vars * num_patches) + v2_expand * num_patches + p_expand
            
            src_nodes_flat = src_nodes.flatten()
            dst_nodes_flat = dst_nodes.flatten()
            
            edge_list.append(torch.stack([src_nodes_flat, dst_nodes_flat], dim=0))
            edge_type_list.append(torch.ones(src_nodes_flat.shape[0], dtype=torch.long, device=device))
        
        # Type 2: Lagged influence
        if num_patches > 1:
            b_expand = b_idx.view(-1, 1, 1, 1).expand(batch_size, num_vars, num_vars, num_patches-1)
            v_src_expand = v_idx.view(1, 1, -1, 1).expand(batch_size, num_vars, num_vars, num_patches-1)
            v_dst_expand = v_idx.view(1, -1, 1, 1).expand(batch_size, num_vars, num_vars, num_patches-1)
            p_expand = p_idx[1:].view(1, 1, 1, -1).expand(batch_size, num_vars, num_vars, num_patches-1)
            
            src_nodes = b_expand * (num_vars * num_patches) + v_src_expand * num_patches + (p_expand - 1)
            dst_nodes = b_expand * (num_vars * num_patches) + v_dst_expand * num_patches + p_expand
            
            src_nodes_flat = src_nodes.flatten()
            dst_nodes_flat = dst_nodes.flatten()
            
            edge_list.append(torch.stack([src_nodes_flat, dst_nodes_flat], dim=0))
            edge_type_list.append(torch.full((src_nodes_flat.shape[0],), 2, dtype=torch.long, device=device))
        
        if len(edge_list) > 0:
            edge_index = torch.cat(edge_list, dim=1)
            edge_type = torch.cat(edge_type_list, dim=0)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_type = torch.empty((0,), dtype=torch.long, device=device)
        
        return edge_index, edge_type
    
    def add_similarity_edges(self, x, edge_index, edge_type, batch_size, num_vars, num_patches):
        """Add top-k similarity edges based on exact cosine similarity"""
        if self.k_sim <= 0:
            return edge_index, edge_type
        
        device = x.device
        N = x.size(0)
        
        x_norm = F.normalize(x, p=2, dim=-1)
        sim_matrix = torch.matmul(x_norm, x_norm.t())
        
        sim_matrix.fill_diagonal_(-float('inf'))
        
        _, top_k_indices = torch.topk(sim_matrix, k=min(self.k_sim, N-1), dim=1)
        src_nodes = torch.arange(N, device=device).view(-1, 1).expand(-1, top_k_indices.size(1))
        dst_nodes = top_k_indices
        src_nodes_flat = src_nodes.flatten()
        dst_nodes_flat = dst_nodes.flatten()
        
        new_edges = torch.stack([src_nodes_flat, dst_nodes_flat], dim=0)
        new_edge_types = torch.full((src_nodes_flat.shape[0],), 3, dtype=torch.long, device=device)
        
        edge_index = torch.cat([edge_index, new_edges], dim=1)
        edge_type = torch.cat([edge_type, new_edge_types])
        
        return edge_index, edge_type
    
    def forward(self, x, batch_size, num_vars, num_patches):
        x_flat = x.reshape(-1, self.d_model)
        
        edge_index, edge_type = self.build_patch_graph(batch_size, num_vars, num_patches, x.device)
        edge_index, edge_type = self.add_similarity_edges(x_flat, edge_index, edge_type, 
                                                          batch_size, num_vars, num_patches)
        
        h = x_flat
        for layer in self.gnn_layers_list:
            h = layer(h, edge_index, edge_type)
            h = self.dropout(h)
        
        h = self.layer_norm(h)
        
        gate = torch.sigmoid(self.gate)
        h = gate * h + (1 - gate) * x_flat
        
        h = h.view(batch_size * num_vars, num_patches, self.d_model)
        
        return h


class GPT2withHPRG(GPT2Model):
    """GPT2 model with HPRG integration"""
    def __init__(self, config, args):
        super().__init__(config)
        self.n_layer = config.n_layer
        self.d_model = config.hidden_size
        self.gnn_layer_index = args.gnn_layer_index
        
        self.prg = HPRG(
            d_model=self.d_model,
            gnn_hidden=args.gnn_hidden_prg,
            gnn_layers=args.gnn_layers_prg,
            k_sim=args.k_sim,
            gate_init=args.gate_init_prg,
            num_relations=4,
            dropout=args.dropout
        )
            
        self.batch_size = None
        self.num_vars = None
        self.num_patches = None

    def set_batch_info(self, batch_size, num_vars, num_patches):
        """Set batch information for PRG computation"""
        self.batch_size = batch_size
        self.num_vars = num_vars
        self.num_patches = num_patches

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Cross-attention mask
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
                    
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if i in self.gnn_layer_index and self.batch_size is not None:
                hidden_states_reshaped = hidden_states.view(
                    self.batch_size * self.num_vars, self.num_patches, self.d_model
                )
                
                # Apply PRG
                prg_output = self.prg(
                    hidden_states_reshaped, 
                    self.batch_size, 
                    self.num_vars, 
                    self.num_patches
                )
                
                prg_output = prg_output.view(batch_size, -1, self.d_model)
                

                hidden_states = prg_output

            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
                    
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]   
            
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        if len(self.h) in self.gnn_layer_index and self.batch_size is not None:
            hidden_states_reshaped = hidden_states.view(
                self.batch_size * self.num_vars, self.num_patches, self.d_model
            )
            prg_output = self.prg(
                hidden_states_reshaped, 
                self.batch_size, 
                self.num_vars, 
                self.num_patches
            )
            prg_output = prg_output.view(batch_size, -1, self.d_model)
            hidden_states = prg_output
                
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class GraFT(nn.Module):    
    def __init__(self, configs, device="cuda:0"):
        super(GraFT, self).__init__()
        self.device = device
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        # PRG parameters # change and check last 
        self.use_temporal = getattr(configs, 'use_temporal', True)
        self.use_cross_var = getattr(configs, 'use_cross_var', True)
        self.use_lag = getattr(configs, 'use_lag', True)
        self.k_sim = getattr(configs, 'k_sim', 5)
        self.gnn_layers_prg = getattr(configs, 'gnn_layers_prg', 2)
        self.gnn_hidden_prg = getattr(configs, 'gnn_hidden_prg', configs.d_model)
        self.gate_init_prg = getattr(configs, 'gate_init_prg', 0.5)
        
        if configs.is_gpt:
            if configs.pretrain:
                config = GPT2Config.from_pretrained('gpt2')
                config.n_layer = configs.gpt_layers
                
                if hasattr(configs, 'd_model') and configs.d_model != config.hidden_size:
                    print(f"Warning: configs.d_model ({configs.d_model}) doesn't match GPT2 hidden_size ({config.hidden_size}). Using GPT2 hidden_size.")
                configs.d_model = config.hidden_size
                
                self.gpt2 = GPT2withHPRG.from_pretrained('gpt2', config=config, args=configs)

                if configs.freeze and configs.pretrain:
                    for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                        if 'ln' in name or 'wpe' in name:
                            param.requires_grad = True
                        elif 'prg' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            else:
                print("------------------no pretrain------------------")
                config = GPT2Config()
                config.n_layer = configs.gpt_layers
                config.hidden_size = configs.d_model
                self.gpt2 = GPT2withHPRG(config, configs)

            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
            
        self.enc_embedding = DataEmbedding_wo_time(
            self.patch_size, 
            configs.d_model,
            configs.embed, 
            configs.freq, 
            configs.in_dropout
        )
        
        self.d_ff = getattr(configs, 'd_ff', configs.d_model)
        self.head_nf = self.d_ff * self.patch_num
        self.output_projection = FlattenHead(
            configs.enc_in, 
            self.head_nf, 
            configs.pred_len,
            head_dropout=configs.out_dropout
        )

        for layer in (self.gpt2, self.enc_embedding, self.output_projection):
            layer.to(device=device)
            layer.train()
        
        self.revin_flag = configs.revin_flag
        if self.revin_flag == 1:
            self.revin_layer = RevIN(1).to(device)

        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.c_out = getattr(configs, 'c_out', configs.enc_in)

    def forward(self, x, ii=None):
        B, L, M = x.shape
        
        if self.revin_flag == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
            x /= stdev

        x = x.permute(0, 2, 1).contiguous()
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        enc_out = self.enc_embedding(x)
        
        if self.is_gpt:
            self.gpt2.set_batch_info(B, M, self.patch_num)
            # Apply transformer with PRG
            outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        else:
            outputs = enc_out

        outputs = outputs[:, :, :self.d_ff]
        outputs = torch.reshape(
            outputs, (-1, M, outputs.shape[-2], outputs.shape[-1]))
        outputs = outputs.permute(0, 1, 3, 2).contiguous()
        
        outputs = self.output_projection(outputs)
        outputs = outputs.permute(0, 2, 1).contiguous()

        if self.revin_flag == 1:
            outputs = self.revin_layer(outputs, 'denorm')
        else:
            outputs = outputs * stdev
            outputs = outputs + means

        return outputs