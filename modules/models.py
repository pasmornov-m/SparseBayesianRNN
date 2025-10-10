import torch
import torch.nn as nn
import math
from modules import utils


class LSTM(nn.Module):
    def __init__(self,
                 incoming,
                 num_units,
                 ingate=None,
                 forgetgate=None,
                 cell=None,
                 outgate=None,
                 hid_init=0.0,
                 cell_init=0.0,
                 learn_init=True,
                 nonlinearity=torch.tanh,
                 backwards=False,
                 gradient_steps=-1,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop=False):
        
        incomings = incoming if hid_prop else [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        super().__init__()

        self.nonlinearity = nonlinearity
        self.num_units = num_units
        
        if isinstance(incoming, int):
            self.num_inputs = incoming
        elif isinstance(incoming, (tuple, list)):
            self.num_inputs = int(math.prod(incoming[2:])) if len(incoming) > 2 else incoming[-1]
        else:
            raise ValueError("incoming must be int or shape-like (tuple/list)")
        
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.only_return_final = only_return_final
        self.hidden_noise = torch.ones(1, dtype=torch.float32)
        self.hidden_clip = torch.ones(1, dtype=torch.float32)
        self.mu_hid = torch.ones(1, dtype=torch.float32)
        self.log_sigma2_hid = torch.ones(1, dtype=torch.float32)
        self.learn_init = learn_init
        self.hid_prop = hid_prop

        if ingate is None:
            self.W_in_to_ingate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_ingate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_ingate = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_ingate) # GlorotUniform
            nn.init.orthogonal_(self.W_hid_to_ingate, gain=1.1) # Orthogonal
            self.nonlinearity_ingate = utils.hard_sigmoid
        
        if forgetgate is None:
            self.W_in_to_forgetgate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_forgetgate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_forgetgate = nn.Parameter(torch.full((self.num_units,), 1.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_forgetgate)
            nn.init.orthogonal_(self.W_hid_to_forgetgate, gain=1.1)
            self.nonlinearity_forgetgate = utils.hard_sigmoid

        if cell is None:
            self.W_in_to_cell = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_cell = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_cell = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_cell)
            nn.init.orthogonal_(self.W_hid_to_cell, gain=1.1)
            self.nonlinearity_cell = torch.tanh

        if outgate is None:
            self.W_in_to_outgate = nn.Parameter(torch.empty(self.num_inputs, self.num_units))
            self.W_hid_to_outgate = nn.Parameter(torch.empty(self.num_units, self.num_units))
            self.b_outgate = nn.Parameter(torch.full((self.num_units,), 0.0, dtype=torch.float32))
            nn.init.xavier_uniform_(self.W_in_to_outgate)
            nn.init.orthogonal_(self.W_hid_to_outgate, gain=1.1)
            self.nonlinearity_outgate = utils.hard_sigmoid

        self.hid_init = nn.Parameter(
            torch.full((1, self.num_units), hid_init, dtype=torch.float32),
            requires_grad=learn_init
        )

        self.cell_init = nn.Parameter(
            torch.full((1, self.num_units), cell_init, dtype=torch.float32),
            requires_grad=learn_init
        )
    
    def input_preactivation(self, input: torch.Tensor, gate_type: str) -> torch.Tensor:
        if gate_type == 'input':
            return input @ self.W_in_to_ingate
        elif gate_type == 'forget':
            return input @ self.W_in_to_forgetgate
        elif gate_type == 'cell':
            return input @ self.W_in_to_cell
        elif gate_type == 'output':
            return input @ self.W_in_to_outgate
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def get_output_shape_for(self, input_shape):
        """
        input_shape: tuple, (batch_size, seq_len, input_size)
        """
        batch_size, seq_len = input_shape[0], input_shape[1]

        if self.only_return_final:
            return (batch_size, self.num_units)
        elif self.hid_prop:
            return (2, batch_size, seq_len, self.num_units)
        else:
            return (batch_size, seq_len, self.num_units)
    
    def generate_noise_and_clip(self, num_batch, seq_len):
        return
    
    def forward(self, inputs, deterministic: bool = False, clip: bool = False):
        """
        PyTorch forward, сохраняет логику оригинального get_output_for.
        inputs: либо тензор (batch, seq_len, input_dim) либо список/tuple, где inputs[0] - вход,
                и при self.mask_incoming_index > 0 mask находится в inputs[self.mask_incoming_index].
        Возвращает:
        - если self.only_return_final: (batch, num_units)
        - elif self.hid_prop: (2, batch, seq_len, num_units)
        - else: (batch, seq_len, num_units)
        """
        if isinstance(inputs, (list, tuple)):
            input = inputs[0]
        else:
            input = inputs

        mask = None
        if getattr(self, "mask_incoming_index", -1) > 0:
            if isinstance(inputs, (list, tuple)) and len(inputs) > self.mask_incoming_index:
                mask = inputs[self.mask_incoming_index]
            else:
                mask = None

        input = input.transpose(0, 1)
        seq_len, num_batch, _ = input.shape

        try:
            self.generate_noise_and_clip(num_batch, deterministic, clip)
        except TypeError:
            self.generate_noise_and_clip(num_batch, seq_len)

        input_i = self.input_preactivation(input, 'input', deterministic=deterministic, clip=clip) + self.b_ingate
        input_f = self.input_preactivation(input, 'forget', deterministic=deterministic, clip=clip) + self.b_forgetgate
        input_c = self.input_preactivation(input, 'cell', deterministic=deterministic, clip=clip) + self.b_cell
        input_o = self.input_preactivation(input, 'output', deterministic=deterministic, clip=clip) + self.b_outgate

        if self.hid_prop:
            hid_init, cell_init = inputs[1][0], inputs[1][1]
        else:
            hid_init = self.hid_init.expand(num_batch, -1).to(input.device, dtype=input.dtype)
            cell_init = self.cell_init.expand(num_batch, -1).to(input.device, dtype=input.dtype)

        hid = hid_init
        cell = cell_init

        hid_seq = []
        cell_seq = []

        if mask is not None:
            if mask.ndim == 3:
                mask_seq = mask.permute(1, 0, 2)
            else:
                mask_seq = mask.unsqueeze(-1).permute(1, 0, 2)
            mask_seq = mask_seq.to(dtype=input.dtype, device=input.device)
        else:
            mask_seq = None

        for t in range(seq_len):
            input_n_i = input_i[t]
            input_n_f = input_f[t]
            input_n_c = input_c[t]
            input_n_o = input_o[t]

            hid_preact_i = self.hidden_preactivation(hid, 'input')
            hid_preact_f = self.hidden_preactivation(hid, 'forget')
            hid_preact_c = self.hidden_preactivation(hid, 'cell')
            hid_preact_o = self.hidden_preactivation(hid, 'output')

            ingate = self.nonlinearity_ingate(input_n_i + hid_preact_i)
            forgetgate = self.nonlinearity_forgetgate(input_n_f + hid_preact_f)
            cell_candidate = self.nonlinearity_cell(input_n_c + hid_preact_c)
            cell = forgetgate * cell + ingate * cell_candidate
            outgate = self.nonlinearity_outgate(input_n_o + hid_preact_o)
            hid = outgate * self.nonlinearity(cell)

            if mask_seq is not None:
                m = mask_seq[t]
                cell = m * cell + (1.0 - m) * cell_init
                hid  = m * hid  + (1.0 - m) * hid_init

            hid_seq.append(hid)
            cell_seq.append(cell)

        cell_out = torch.stack(cell_seq, dim=0)
        hid_out = torch.stack(hid_seq, dim=0)

        if self.only_return_final:
            return hid_out[-1]

        if self.backwards:
            hid_out = hid_out.flip(dims=[0])
            cell_out = cell_out.flip(dims=[0])

        hid_out = hid_out.transpose(0, 1)
        cell_out = cell_out.transpose(0, 1)

        if self.hid_prop:
            return torch.cat([hid_out.unsqueeze(0), cell_out.unsqueeze(0)], dim=0)
        else:
            return hid_out


class BayesianLSTM(LSTM):
    """
    config: L probabilistic weight with lognormal prior, N probabilistic weight with standart normal prior, 
           D deterministic learnable weight, 
           C constant weight (1 for multiplicative and 0 for additive weights)\
           
           config[0]: W input_to_hidden, W hidden_to_hidden
                     L N D (C is not supported)
           config[1]: hat Z preactivation multiplicative weights
                     L N D C
           config[2]: Z input and hidden multiplicative weights
                     L N D C I R
    """
    def __init__(self, 
                 incoming, 
                 num_units,
                 log_sigma_in_init = -3.0, 
                 log_sigma_hid_init = -3.0,
                 ingate=None,
                 forgetgate=None,
                 cell=None,
                 outgate=None,
                 hid_init=0.0,
                 cell_init=0.0,
                 learn_init=True,
                 nonlinearity=torch.tanh,
                 backwards=False,
                 gradient_steps=-1,
                 mask_input=None,
                 only_return_final=False,
                 hid_prop = False,
                 config="DCC"):
 
        super().__init__(incoming, 
                         num_units, 
                         ingate, 
                         forgetgate, 
                         cell, 
                         outgate,
                         hid_init, 
                         cell_init, 
                         learn_init, 
                         nonlinearity, 
                         backwards, 
                         gradient_steps, 
                         mask_input,
                         only_return_final,
                         hid_prop)
        
        self.reg = True
        self.config = config
        self.log_sigma_in_init = log_sigma_in_init
        self.log_sigma_hid_init = log_sigma_hid_init
        self.dtype = torch.float32

        if self.config[0] in {"L", "N"}:
            self.logsig_w_in = nn.Parameter(torch.full((4, incoming, num_units), log_sigma_in_init))
            self.logsig_w_hid = nn.Parameter(torch.full((4, num_units, num_units), log_sigma_hid_init))
        else:
            self.register_buffer("logsig_w_in", torch.zeros(4))
            self.register_buffer("logsig_w_hid", torch.zeros(4))
        
        if self.config[2] in {"L", "N", "D", "I"}:
            self.mu_in = nn.Parameter(torch.ones(incoming))
        if self.config[2] in {"L", "N", "I"}:
            self.logsig_in = nn.Parameter(torch.full((incoming,), log_sigma_in_init))
        if self.config[2] in {"L", "N", "D", "R"}:
            self.mu_hid = nn.Parameter(torch.ones(num_units))
        if self.config[2] in {"L", "N", "R"}:
            self.logsig_hid = nn.Parameter(torch.full((num_units,), log_sigma_hid_init))

        if self.config[1] in {"L", "N", "D"}:
            self.mu_gates = nn.Parameter(torch.ones(4, num_units))
        if self.config[1] in {"L", "N"}:
            self.logsig_gates = nn.Parameter(torch.full((4, num_units), log_sigma_hid_init))

        self.input_noise = None
        self.hidden_noise = None
        self.input_clip = None
        self.hidden_clip = None

        self.thresh = 3.0
    
    def generate_noise_and_clip(self, num_batch, deterministic=False, clip=False):
        if not deterministic:
            # --- W_in, W_hid noise ---
            if self.config[0] in {"L", "N"}:
                self.input_w_noise = torch.randn(4, self.num_inputs, self.num_units) * torch.exp(self.logsig_w_in)
                self.hidden_w_noise = torch.randn(4, self.num_units, self.num_units) * torch.exp(self.logsig_w_hid)
            else:
                self.input_w_noise = torch.zeros(4)
                self.hidden_w_noise = torch.zeros(4)

            # --- input/hidden noise ---
            if self.config[2] in {"L", "N"}:
                self.input_noise = torch.randn(num_batch, self.num_inputs) * torch.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = torch.randn(num_batch, self.num_units) * torch.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = torch.randn(num_batch, self.num_inputs) * torch.exp(self.logsig_in) + self.mu_in
                self.hidden_noise = torch.ones(1)
            elif self.config[2] == "R":
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.randn(num_batch, self.num_units) * torch.exp(self.logsig_hid) + self.mu_hid
            elif self.config[2] == "D":
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.ones(1)
        
            # --- gates noise ---
            if self.config[1] in {"L", "N"}:
                self.gates_noise = torch.randn(4, num_batch, self.num_units) * torch.exp(self.logsig_gates)[:, None, :] + self.mu_gates[:, None, :]
            elif self.config[1] == "D":
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = torch.ones(4)
        
        else:
            self.input_w_noise = torch.zeros(4)
            self.hidden_w_noise = torch.zeros(4)

            if self.config[2] in {"L", "N", "D"}:
                self.input_noise = self.mu_in
                self.hidden_noise = self.mu_hid
            elif self.config[2] == "I":
                self.input_noise = self.mu_in
                self.hidden_noise = torch.ones(1)
            elif self.config[2] == "R":
                self.input_noise = torch.ones(1)
                self.hidden_noise = self.mu_hid
            else:
                self.input_noise = torch.ones(1)
                self.hidden_noise = torch.ones(1)

            if self.config[1] in {"L", "N", "D"}:
                self.gates_noise = self.mu_gates
            else:
                self.gates_noise = torch.ones(4, dtype=self.dtype)
        
        if clip:
            if self.config[0] == "L":
                W_in_cat = torch.cat([self.W_in_to_ingate[None,:,:],
                                    self.W_in_to_forgetgate[None,:,:],
                                    self.W_in_to_cell[None,:,:],
                                    self.W_in_to_outgate[None,:,:]], dim=0)
                log_alpha_w_in = utils.clip_func(2*self.logsig_w_in - torch.log(W_in_cat**2 + self.epsilon))
                self.input_w_clip = log_alpha_w_in <= self.thresh

                W_hid_cat = torch.cat([self.W_hid_to_ingate[None,:,:],
                                    self.W_hid_to_forgetgate[None,:,:],
                                    self.W_hid_to_cell[None,:,:],
                                    self.W_hid_to_outgate[None,:,:]], dim=0)
                log_alpha_w_hid = utils.clip_func(2*self.logsig_w_hid - utils.safe_torch_log(W_hid_cat**2))
                self.hidden_w_clip = log_alpha_w_hid <= self.thresh
            else:
                self.input_w_clip = torch.ones(4)
                self.hidden_w_clip = torch.ones(4)

            if self.config[2] == "L":
                log_alpha_in = utils.clip_func(2*self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                self.input_clip = log_alpha_in <= self.thresh
                log_alpha_hid = utils.clip_func(2*self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                self.hidden_clip = log_alpha_hid <= self.thresh
            elif self.config[2] == "I":
                log_alpha_in = utils.clip_func(2*self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                self.input_clip = log_alpha_in <= self.thresh
                self.hidden_clip = torch.ones(1)
            elif self.config[2] == "R":
                self.input_clip = torch.ones(1)
                log_alpha_hid = utils.clip_func(2*self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                self.hidden_clip = log_alpha_hid <= self.thresh
            else:
                self.input_clip = torch.ones(1)
                self.hidden_clip = torch.ones(1)

            if self.config[1] == "L":
                log_alpha_gates = utils.clip_func(2*self.logsig_gates - utils.safe_torch_log(self.mu_gates**2))
                self.gates_clip = log_alpha_gates <= self.thresh
            else:
                self.gates_clip = torch.ones(4)
        
        else:
            self.input_w_clip = torch.ones(4)
            self.hidden_w_clip = torch.ones(4)
            self.input_clip = torch.ones(1)
            self.hidden_clip = torch.ones(1)
            self.gates_clip = torch.ones(4)
        
        self.W_hid = torch.cat([
            self.W_hid_to_ingate + self.hidden_w_noise[0],
            self.W_hid_to_forgetgate + self.hidden_w_noise[1],
            self.W_hid_to_cell + self.hidden_w_noise[2],
            self.W_hid_to_outgate + self.hidden_w_noise[3]
        ], dim=1)

        self.W_in = torch.cat([
            self.W_in_to_ingate + self.input_w_noise[0],
            self.W_in_to_forgetgate + self.input_w_noise[1],
            self.W_in_to_cell + self.input_w_noise[2],
            self.W_in_to_outgate + self.input_w_noise[3],
        ], dim=1)

        return
    
    def eval_reg(self, train_size):
        W_in = torch.cat([
            self.W_in_to_ingate.unsqueeze(0),
            self.W_in_to_forgetgate.unsqueeze(0),
            self.W_in_to_cell.unsqueeze(0),
            self.W_in_to_outgate.unsqueeze(0)
        ], dim=0)

        if self.config[0] == "N":
            KL_element_in = -self.logsig_w_in + 0.5 * (torch.exp(2 * self.logsig_w_in) + W_in**2) - 0.5
            KL = KL_element_in.sum()
        elif self.config[0] == "L":
            log_alpha_w_in = utils.clip_func(2 * self.logsig_w_in - utils.safe_torch_log(W_in**2))
            KL = utils.alpha_regf(log_alpha_w_in).sum()
        else:
            KL = torch.zeros(1, dtype=self.dtype, device=self.logsig_w_in.device).sum()

        # Скрытые веса
        W_hid = torch.cat([
            self.W_hid_to_ingate.unsqueeze(0),
            self.W_hid_to_forgetgate.unsqueeze(0),
            self.W_hid_to_cell.unsqueeze(0),
            self.W_hid_to_outgate.unsqueeze(0)
        ], dim=0)

        if self.config[0] == "N":
            KL_element_hid = -self.logsig_w_hid + 0.5 * (torch.exp(2 * self.logsig_w_hid) + W_hid**2) - 0.5
            KL += KL_element_hid.sum()
        elif self.config[0] == "L":
            log_alpha_w_hid = utils.clip_func(2 * self.logsig_w_hid - utils.safe_torch_log(W_hid**2))
            KL += utils.alpha_regf(log_alpha_w_hid).sum()

        # Нейроны
        if self.config[2] in {"L", "R", "I"}:
            if self.config[2] in {"L", "R"}:
                log_alpha_hid = utils.clip_func(2 * self.logsig_hid - utils.safe_torch_log(self.mu_hid**2))
                KL += utils.alpha_regf(log_alpha_hid).sum()
            if self.config[2] in {"L", "I"}:
                log_alpha_in = utils.clip_func(2 * self.logsig_in - utils.safe_torch_log(self.mu_in**2))
                KL += utils.alpha_regf(log_alpha_in).sum()
        elif self.config[2] == "N":
            KL += (-self.logsig_hid + 0.5 * (torch.exp(2 * self.logsig_hid) + self.mu_hid**2) - 0.5).sum()
            KL += (-self.logsig_in + 0.5 * (torch.exp(2 * self.logsig_in) + self.mu_in**2) - 0.5).sum()

        # Гейты
        if self.config[1] == "L":
            log_alpha_gates = utils.clip_func(2 * self.logsig_gates - utils.safe_torch_log(self.mu_gates**2))
            KL += utils.alpha_regf(log_alpha_gates).sum()
        elif self.config[1] == "N":
            KL_element = -self.logsig_gates + 0.5 * (torch.exp(2 * self.logsig_gates) + self.mu_gates**2) - 0.5
            KL += KL_element.sum()
        
        reg = KL / train_size
        return reg
    
    def get_ard(self):
        # --- W ---
        if self.config[0] == "L":
            W_in = torch.cat([
                self.W_in_to_ingate.unsqueeze(0),
                self.W_in_to_forgetgate.unsqueeze(0),
                self.W_in_to_cell.unsqueeze(0),
                self.W_in_to_outgate.unsqueeze(0)
            ], dim=0)
            log_alpha_w_in = 2 * self.logsig_w_in - 2 * utils.safe_torch_log(torch.abs(W_in))
            mask_w_in = log_alpha_w_in < self.thresh

            W_hid = torch.cat([
                self.W_hid_to_ingate.unsqueeze(0),
                self.W_hid_to_forgetgate.unsqueeze(0),
                self.W_hid_to_cell.unsqueeze(0),
                self.W_hid_to_outgate.unsqueeze(0)
            ], dim=0)
            log_alpha_w_hid = 2 * self.logsig_w_hid - 2 * utils.safe_torch_log(torch.abs(W_hid))
            mask_w_hid = log_alpha_w_hid < self.thresh
        else:
            mask_w_in = torch.ones((4,) + self.W_in_to_ingate.shape, dtype=torch.bool, device=self.W_in_to_ingate.device)
            mask_w_hid = torch.ones((4,) + self.W_hid_to_ingate.shape, dtype=torch.bool, device=self.W_hid_to_ingate.device)

        # --- neurons ---
        mask_in = mask_w_in.any(dim=2).any(dim=0)
        mask_hid_by_w = mask_w_hid.any(dim=2).any(dim=0)
        mask_hid_by_z = torch.ones_like(mask_hid_by_w, dtype=torch.bool)
        
        def log_alpha_calc(logsig, mu):
            return 2 * logsig - 2 * utils.safe_torch_log(torch.abs(mu))

        if self.config[2] == "L":
            log_alpha_hid = log_alpha_calc(self.logsig_hid, self.mu_hid)
            log_alpha_in = log_alpha_calc(self.logsig_in, self.mu_in)
            mask_in = torch.logical_and(log_alpha_in < self.thresh, mask_in)
            mask_hid_by_z = log_alpha_hid < self.thresh
        elif self.config[2] == "I":
            log_alpha_in = log_alpha_calc(self.logsig_in, self.mu_in)
            mask_in = torch.logical_and(log_alpha_in < self.thresh, mask_in)
        elif self.config[2] == "R":
            log_alpha_hid = log_alpha_calc(self.logsig_hid, self.mu_hid)
            mask_hid_by_z = log_alpha_hid < self.thresh

        # --- gates ---
        mask = torch.cat([mask_w_in, mask_w_hid], dim=1)
        if self.config[1] == "L":
            log_alpha_gates = log_alpha_calc(self.logsig_gates, self.mu_gates)
            mask_gates = torch.logical_and(log_alpha_gates < self.thresh, mask.any(dim=1))
        else:
            mask_gates = mask.any(dim=1)

        return {
            "w_input": mask_w_in,
            "w_hidden": mask_w_hid,
            "gates": mask_gates,
            "z_input": mask_in,
            "z_hidden_by_w": mask_hid_by_w,
            "z_hidden": mask_hid_by_z,
        }
        
    def prepare_gates_noise(self, num_batch):
        device = getattr(self, "gates_noise", torch.tensor(0)).device
        dtype = self.dtype
        num_units = self.num_units

        # --- gates_noise ---
        if self.gates_noise is None:
            gates_noise = torch.ones((num_batch, 4, num_units), dtype=dtype, device=device)
        else:
            g = self.gates_noise
            if g.ndim == 1:
                gates_noise = g.unsqueeze(0).unsqueeze(-1)
                gates_noise = gates_noise.expand(num_batch, -1, num_units)
            elif g.ndim == 2:
                gates_noise = g.unsqueeze(0).expand(num_batch, -1, -1)
            elif g.ndim == 3:
                gates_noise = g.permute(1, 0, 2).to(dtype=dtype, device=device)
            else:
                raise ValueError(f"Unexpected shape for gates_noise: {g.shape}")

        # --- gates_clip ---
        if self.gates_clip is None:
            gates_clip = torch.ones((1, 4, num_units), dtype=dtype, device=device)
        else:
            gc = self.gates_clip
            if gc.ndim == 0:
                gates_clip = gc.expand(1, 4, num_units)
            elif gc.ndim == 1:
                gates_clip = gc.unsqueeze(0).unsqueeze(-1).expand(1, -1, num_units)
            elif gc.ndim == 2:
                gates_clip = gc.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected shape for gates_clip: {gc.shape}")

        gates_noise_clipped = gates_noise * gates_clip

        return (gates_noise_clipped[:, 0, :], 
                gates_noise_clipped[:, 1, :], 
                gates_noise_clipped[:, 2, :], 
                gates_noise_clipped[:, 3, :])

    def forward(self, inputs, hid_init=None, deterministic: bool = False, clip: bool = False):
        """
        PyTorch forward for BayesianLSTM — повторяет логику get_output_for.
        inputs: либо тензор (batch, seq_len, input_dim) либо список/tuple:
                inputs[0] - input tensor,
                при self.mask_incoming_index > 0 mask ожидается в inputs[self.mask_incoming_index],
                при self.hid_prop ожидается inputs[1] с hid_init и cell_init.
        Возвращает:
        - если self.only_return_final: (batch, num_units)
        - elif self.hid_prop: (2, batch, seq_len, num_units)
        - else: (batch, seq_len, num_units)
        """

        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
        else:
            x = inputs

        mask = None
        if getattr(self, "mask_incoming_index", -1) > 0 and isinstance(inputs, (list, tuple)):
            if len(inputs) > self.mask_incoming_index:
                mask = inputs[self.mask_incoming_index]
        
        if mask is not None:
            if mask.ndim == 3:
                mask_seq = mask
            else:
                mask_seq = mask.unsqueeze(-1)
        else:
            mask_seq = None

        num_batch, seq_len, _ = x.shape
        hid_out = torch.empty((num_batch, seq_len, self.num_units), dtype=self.dtype)
        cell_out = torch.empty_like(hid_out)
 
        self.generate_noise_and_clip(num_batch, deterministic, clip)

        g0_gc0, g1_gc1, g2_gc2, g3_gc3 = self.prepare_gates_noise(num_batch)
        
        x_eff = x * self.input_noise * self.input_clip
        input_preact = torch.matmul(x_eff, self.W_in)
        input_i, input_f, input_c, input_o = torch.chunk(input_preact, 4, dim=-1)

        input_i = input_i * self.input_w_clip[0] + self.b_ingate
        input_f = input_f * self.input_w_clip[1] + self.b_forgetgate
        input_c = input_c * self.input_w_clip[2] + self.b_cell
        input_o = input_o * self.input_w_clip[3] + self.b_outgate
        
        if self.hid_prop and hid_init is not None:
            hid = hid_init[0].to(dtype=self.dtype)
            cell = hid_init[1].to(dtype=self.dtype)
        else:
            hid = None
            cell = None
        
        hn = self.hidden_noise if self.hidden_noise is not None else torch.ones(1, dtype=self.dtype)
        hc = self.hidden_clip if self.hidden_clip is not None else torch.ones(1, dtype=self.dtype)
        hn_hc = hn * hc
        
        t_range = range(seq_len - 1, -1, -1) if self.backwards else range(seq_len)

        for t in t_range:
            cell_prev = cell
            hid_prev = hid
            
            input_n_i = input_i[:, t, :]
            input_n_f = input_f[:, t, :]
            input_n_c = input_c[:, t, :]
            input_n_o = input_o[:, t, :]
            
            hid_preact = hid @ self.W_hid
            hid_preact_i, hid_preact_f, hid_preact_c, hid_preact_o = torch.chunk(hid_preact, 4, dim=1)

            hid_preact_i = hid_preact_i * self.hidden_w_clip[0]
            hid_preact_f = hid_preact_f * self.hidden_w_clip[1]
            hid_preact_c = hid_preact_c * self.hidden_w_clip[2]
            hid_preact_o = hid_preact_o * self.hidden_w_clip[3]

            ingate = self.nonlinearity_ingate((input_n_i + hid_preact_i) * g0_gc0 + self.b_ingate)
            forgetgate = self.nonlinearity_forgetgate((input_n_f + hid_preact_f) * g1_gc1 + self.b_forgetgate)
            cell_candidate = self.nonlinearity_cell((input_n_c + hid_preact_c) * g2_gc2 + self.b_cell)
            
            cell_new = forgetgate * cell_prev + ingate * cell_candidate
            outgate = self.nonlinearity_outgate((input_n_o + hid_preact_o) * g3_gc3 + self.b_outgate)
            hid_new = outgate * self.nonlinearity(cell_new)

            hid_new = hid_new * hn_hc

            if mask_seq is not None:
                m = mask_seq[:, t, :]
                if m.ndim == 2 and m.shape[1] == 1:
                    m = m.expand(-1, self.num_units)
                cell = m * cell_new + (1.0 - m) * cell_prev
                hid = m * hid_new + (1.0 - m) * hid_prev
            else:
                cell = cell_new
                hid = hid_new

            hid_out[:, t, :] = hid
            cell_out[:, t, :] = cell

        if self.only_return_final:
            return hid_out[-1]

        if self.backwards:
            hid_out = hid_out.flip(dims=[1])
            cell_out = cell_out.flip(dims=[1])

        if self.hid_prop:
            return torch.cat([hid_out.unsqueeze(0), cell_out.unsqueeze(0)], dim=0)
        else:
            return hid_out


class Dense(nn.Module):
    def __init__(self, incoming, num_units, nonlinearity=nn.Identity()):
        super(Dense, self).__init__()
        self.num_units = num_units
        self.nonlinearity = nonlinearity
        
        self.W = nn.Parameter(torch.empty(incoming, num_units))
        self.b = nn.Parameter(torch.zeros(num_units))
        
        nn.init.xavier_uniform_(self.W)

    def pre_activation(self, input):
        return torch.matmul(input, self.W)
    
    def get_output_shape_for(self, input_shape):
        return tuple(input_shape[:-1]) + (self.num_units,)

    def get_output_for(self, input):
        return self.nonlinearity(self.pre_activation(input) + self.b)

    def get_ard(self):
        return {"w": torch.ones_like(self.W)}
    
    def forward(self, input):
        """
        input: tensor with last dim == incoming, arbitrary leading dims allowed
        returns: tensor with same leading dims and last dim == num_units
        """
        lin = self.pre_activation(input)
        lin = lin + self.b
        return self.nonlinearity(lin)

class BayesianDense(Dense):
    def __init__(self, 
                 incoming, 
                 num_units, 
                 log_sigma_init = -3.0,
                 W_initializer=None, 
                 b_init=0.0, 
                 nonlinearity=lambda x: x):
        super().__init__(incoming, num_units, nonlinearity)

        if isinstance(incoming, int):
            self.num_inputs = int(incoming)
        else:
            try:
                self.num_inputs = int(incoming[-1])
            except Exception:
                raise ValueError("incoming must be int or shape-like")

        self.num_units = int(num_units)
        self.nonlinearity = nonlinearity
        self.thresh = 3.0
        self.dtype = torch.float32

        self.W = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=self.dtype))
        self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=self.dtype))
        self.log_sigma = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=self.dtype))

        if W_initializer is None:
            nn.init.xavier_uniform_(self.W)
        else:
            W_initializer(self.W)

    def pre_activation(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False):
        """
        input: либо 2D (batch, input_dim) либо 3D (batch, seq_len, input_dim)
        Возвращает: mu + шум*si (или только mu в deterministic режиме)
        """
        W_eff = self.W
        sigma2 = torch.exp(2.0 * self.log_sigma)

        if clip:
            log_alpha = utils.clip_func(2.0 * self.log_sigma - utils.safe_torch_log(W_eff.pow(2)))
            clip_mask = log_alpha.ge(self.thresh)
            W_eff = torch.where(clip_mask, torch.zeros_like(W_eff), W_eff)
            sigma2 = torch.where(clip_mask, torch.zeros_like(sigma2), sigma2)

        if deterministic:
            return input @ W_eff

        mu = input @ W_eff
        si = torch.sqrt((input * input) @ sigma2 + 1e-8)

        if input.ndim == 2:
            noise = torch.randn_like(mu)
        else:
            noise = torch.randn((mu.shape[0], 1, mu.shape[2]))

        return mu + noise * si
        

    def eval_reg(self, train_size: float):
        """
        alpha regularization: utils.alpha_regf(clip_func(2*log_sigma - log(W^2))).sum() / train_size
        Возвращаем torch scalar
        """
        log_alpha = utils.clip_func(2.0 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
        reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
        return reg

    def get_ard(self) -> dict[str, torch.Tensor]:
        """
        Возвращаем torch-маску (bool tensor).
        Маска не требует градиентов и используется для sparsification.
        """
        W = self.W.detach()
        log_sigma = self.log_sigma.detach()
        log_alpha = 2.0 * log_sigma - 2.0 * utils.safe_torch_log(torch.abs(W))
        mask = (log_alpha < self.thresh)

        return {"w": mask}
    
    def forward(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False, **kwargs):
        """
        input: tensor with last dim == num_inputs, can be 2D (batch, in) or 3D (batch, seq_len, in)
        deterministic, clip: передаются в pre_activation и управляют режимом
        Возвращает: nonlinearity( pre_activation(input, deterministic, clip) + b )
        """
        out = self.pre_activation(input, deterministic=deterministic, clip=clip)
        out = out + self.b
        return self.nonlinearity(out)


class BayesianDense_noLRT(Dense):
    def __init__(self, incoming, num_units, log_sigma_init=-3.0,
                 W_initializer=None, b_init=0.0, nonlinearity=lambda x: x):
        """
        incoming: int (input size) or shape-like with last dim = input size.
        """
        super().__init__(incoming, num_units, nonlinearity)
        if isinstance(incoming, int):
            self.num_inputs = int(incoming)
        else:
            try:
                self.num_inputs = int(incoming[-1])
            except Exception:
                raise ValueError("incoming must be int or shape-like")

        self.num_units = int(num_units)
        self.nonlinearity = nonlinearity
        self.thresh = 3.0

        self.W = nn.Parameter(torch.empty(self.num_inputs, self.num_units, dtype=torch.float32))
        self.b = nn.Parameter(torch.full((self.num_units,), float(b_init), dtype=torch.float32))
        self.log_sigma = nn.Parameter(torch.full((self.num_inputs, self.num_units), float(log_sigma_init), dtype=torch.float32))

        if W_initializer is None:
            nn.init.xavier_uniform_(self.W)
        else:
            try:
                W_initializer(self.W)
            except Exception:
                self.W.data.copy_(torch.tensor(W_initializer(self.W.shape), dtype=self.W.dtype))

    def pre_activation(self, input: torch.Tensor, deterministic: bool = False, clip: bool = False):
        """
        input: 2D (batch, in) или 3D (batch, seq_len, in) (и др. формы с последним измерением in)
        """
        sigma2 = torch.exp(2 * self.log_sigma)
        W_eff = self.W
        
        if clip:
            log_alpha = utils.clip_func(2 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
            clip_mask = log_alpha.ge(self.thresh)
            W_eff = torch.where(clip_mask, torch.zeros_like(W_eff), W_eff)
            sigma2 = torch.where(clip_mask, torch.zeros_like(sigma2), sigma2)

        if deterministic:
            return input @ W_eff
        
        if input.ndim == 2:
            mu = input @ W_eff
            si = torch.sqrt(input.pow(2) @ sigma2 + 1e-8)
            return mu + torch.randn_like(mu) * si
        else:
            W_noisy = W_eff + torch.randn_like(W_eff) * torch.exp(self.log_sigma)
            return input @ W_noisy

    def eval_reg(self, train_size: float):
        log_alpha = utils.clip_func(2 * self.log_sigma - utils.safe_torch_log(self.W.pow(2)))
        reg = utils.alpha_regf(log_alpha).sum() / float(train_size)
        return reg
    
    def get_ard(self) -> dict[str, torch.Tensor]:
        """
        Возвращаем torch-маску (bool tensor).
        Маска не требует градиентов и используется для sparsification.
        """
        W = self.W.detach()
        log_sigma = self.log_sigma.detach()
        log_alpha = 2 * log_sigma - 2 * utils.safe_torch_log(torch.abs(W))
        mask = (log_alpha < self.thresh)
        return {"w": mask}
    
    def forward(self, input: torch.Tensor, deterministic: bool = None, clip: bool = False) -> torch.Tensor:
        """
        input: tensor with last dim == num_inputs, can be 2D (batch, in) or 3D (batch, seq_len, in)
        deterministic: if None -> deterministic = not self.training (eval mode); else use explicit value
        clip: whether to apply clipping logic inside pre_activation
        """
        if deterministic is None:
            deterministic = not self.training
        out = self.pre_activation(input, deterministic=deterministic, clip=clip)
        out = out + self.b
        return self.nonlinearity(out)


class LMNet(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            n_hidden, 
            config, 
            hid_prop=False, 
            batch_size=32,
            device='cpu'):
        """
        vocab_size: размер словаря
        n_hidden: размер скрытого состояния LSTM
        config: строка конфигурации (как в оригинале)
        hid_prop: если True — предусмотрена возможность прокинуть hid_init
        batch_size: использовался в оригинале при создании shared hid (32)
        device: 'cpu' или 'cuda'
        """
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_hidden = int(n_hidden)
        self.config = config
        self.hid_prop = bool(hid_prop)
        self.batch_size = int(batch_size)
        self.device = device

        # LSTM: принимает вход feature dim = vocab_size
        self.lstm = BayesianLSTM(
            incoming=self.vocab_size,
            num_units=self.n_hidden,
            config=config[:3],
            only_return_final=False,
            learn_init=False,
            hid_prop=hid_prop)

        if self.hid_prop:
            hid_tensor = torch.zeros((2, self.batch_size, self.n_hidden), dtype=torch.float32)
            self.register_buffer('hid', hid_tensor)

        # dense: если hid_prop, в оригинале Dense применялся к определенному срезу LSTM-выхода;
        # здесь реализуем Dense, который можно вызвать на тензоре формы (batch, seq_len, n_hidden).
        if self.hid_prop:
            # создаём Dense, который применяют к LSTM-выходу по временным шагам.
            DenseClass = BayesianDense_noLRT if config[-1] == "L" else Dense
            self.dense = DenseClass(incoming=self.n_hidden, num_units=self.vocab_size)
        else:
            DenseClass = BayesianDense if config[-1] == "L" else Dense
            self.dense = DenseClass(incoming=self.n_hidden, num_units=self.vocab_size)

    def forward(self, inp, use_hid_init=False):
        """
        inp: LongTensor (batch, seq_len) — индексы слов (как T.imatrix)
        use_hid_init: если True и hid_prop=True — использовать self.hid как hid_init
        Возвращает: распределения (batch*seq_len, vocab_size) или (batch, seq_len, vocab_size) в зависимости от нужд.
        """
        if self.hid_prop:
            b = inp.size(0)
            num_layers = 2

            if hasattr(self, "hid_next") and self.hid_next is not None:
                if self.hid_next.size(1) != b:
                    if self.hid_next.size(1) > b:
                        hid_next = self.hid_next[:, :b, :].clone()
                    else:
                        pad = torch.zeros((self.hid_next.size(0), b - self.hid_next.size(1), self.n_hidden), dtype=self.hid_next.dtype)
                        hid_next = torch.cat([self.hid_next, pad], dim=1)
                else:
                    hid_next = self.hid_next
                hid_init = hid_next.clone()
            else:
                hid_init = torch.zeros((num_layers, b, self.n_hidden))

            lstm_out = self.lstm(inp, hid_init=hid_init)
            last_hidden = lstm_out[:, :, -1, :].detach()
            self.hid_next = last_hidden.clone()
        else:
            lstm_out = self.lstm(inp)

        hid_out = lstm_out[0]
        logits = self.dense(hid_out)
            
        return logits

    def compute_compression_masks(self):
        """
        Возвращает маски компрессии в виде torch.Tensor (на device модели).
        Маски отвязаны от графа вычислений (detach).
        """
        device = next(self.parameters()).device

        masks_lstm = self.lstm.get_ard()
        masks_dense = self.dense.get_ard()

        mask_vocabulary = masks_lstm["z_input"].to(dtype=torch.bool, device=device)

        mask_hidden = torch.logical_or(
            masks_lstm["z_hidden_by_w"],
            masks_dense["w"].any(dim=1)
        )
        mask_hidden = torch.logical_and(mask_hidden, masks_lstm["z_hidden"])
        mask_hidden = mask_hidden.to(dtype=torch.bool, device=device)

        w_in = masks_lstm["w_input"].clone().to(device)
        w_hid = masks_lstm["w_hidden"].clone().to(device)
        w_dense = masks_dense["w"].clone().to(device)
        gates = masks_lstm["gates"].clone().to(device)

        w_in[:, ~mask_vocabulary, :] = 0
        w_in[:, :, ~mask_hidden] = 0
        w_hid[:, ~mask_hidden, :] = 0
        w_hid[:, :, ~mask_hidden] = 0
        w_dense[~mask_hidden] = 0
        gates[:, ~mask_hidden] = 0

        return mask_vocabulary, mask_hidden, gates, w_in, w_hid, w_dense

    def evaluate_compression(self):
        mask_vocabulary, mask_hidden, mask_gates, mask_w_inp, mask_w_hid, mask_w_dense = self.compute_compression_masks()
        
        w_nonzero, w_all = 0.0, 0.0
        for w in [mask_w_inp, mask_w_hid, mask_w_dense]:
            w_nonzero += float(w.sum())
            w_all += float(w.numel())
        overall_compression = w_all / (w_nonzero + 1e-8)

        print("Compression per layers:")
        for layer_name, masks in [("LSTM", {"z_x":mask_vocabulary, "z_h":mask_hidden, "gates":mask_gates, "w_x":mask_w_inp, "w_h":mask_w_hid}), ("Dense", {"w":mask_w_dense})]:
            print(layer_name, end=": ")
            for key, matrix in masks.items():
                print("(%s: %d/%d)" % (key, int(matrix.sum()), matrix.numel()), end=" ")
            print()
        print(f"Overall compression: {overall_compression:.3f}")
        return overall_compression