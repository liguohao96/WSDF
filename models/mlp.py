import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_in, hidden_channel, num_out, act_fn, last_act_fn=None):
        super().__init__()

        if isinstance(hidden_channel, (tuple, list)):
            pass
        else:
            hidden_channel = [hidden_channel]

        layers = []
        for ci, co in zip(
            [num_in] + hidden_channel,
            hidden_channel + [num_out],
            ):

            linear = nn.Linear(ci, co, bias=True)
            layers.extend([linear, act_fn()])

        del layers[-1]
        if last_act_fn is not None:
            layers.append(last_act_fn())

        # layers = [act_fn()]
        self.args_str = f"num_in={num_in},hidden_channel={hidden_channel},num_out={num_out}"

        self.net = nn.Sequential(*layers)
    
    def extra_repr(self):
        return self.args_str

    def forward(self, x):
        return self.net(x)

    def jacobian(self, x, jac_out):
        NL       = len(self.net)
        inp_list = [x]

        inplace_conf = [None] * NL

        # inp_list[0] = x
        # out = inp_list[0]
        for li in range(NL):
            layer = self.net[li]
            input = inp_list[li]
            # out   = layer(out)
            # inp_list[li+1] = out
            inp_list.append(layer(input))

        def get_jacobian(module, input, output, jac_out):
            # input:   B,P,Q,I
            # jac_out: B,P,Q,O
            # return:  B,P,Q,I
            if isinstance(module, nn.Linear):
                input_shape   = input.shape
                weight        = layer.weight
                # jac_out_shape = input_shape[:-1] + [weight.size(0)]
                if jac_out is None:
                    jac_out = layer.weight.unsqueeze(0).expand(input.size(0),-1,-1) # B,O,I
                    # jac_out = jac_out.reshape(input_shape)                          # B,P,Q,I
                else:
                    jac_out = torch.einsum("b...o,oi->b...i", jac_out, layer.weight)
            elif isinstance(module, nn.ELU):
                # act_drv = torch.where(input > 0, torch.ones_like(input), torch.exp(input))
                act_drv = torch.where(output > 0, torch.ones_like(output), output+1)
                if jac_out is None:
                    # jac_out = torch.einsum("bi->bii", act_drv)
                    jac_out = torch.diag_embed(act_drv)
                else:
                    for _ in range(jac_out.dim() - act_drv.dim()):
                        act_drv = act_drv.unsqueeze(1)
                    jac_out = act_drv * jac_out
                    # jac_out = torch.einsum("b...i,b...i->b", act_drv, jac_out) # [B, Ci] [B, O, Ci]
                    # act_drv * jac_out
            elif isinstance(module, nn.ReLU):
                act_drv = torch.where(output > 0, torch.ones_like(output), torch.zeros_like(output))
                if jac_out is None:
                    jac_out = torch.diag_embed(act_drv)
                else:
                    for _ in range(jac_out.dim() - act_drv.dim()):
                        act_drv = act_drv.unsqueeze(1)
                    jac_out = act_drv * jac_out
            elif isinstance(module, nn.Softplus):
                beta, threshold = module.beta, module.threshold
                act_drv = torch.where(input*beta>threshold, 1, torch.exp((input-output)*beta))
                if jac_out is None:
                    jac_out = torch.diag_embed(act_drv)
                else:
                    for _ in range(jac_out.dim() - act_drv.dim()):
                        act_drv = act_drv.unsqueeze(1)
                    jac_out = act_drv * jac_out
            elif hasattr(layer, "jacobian"):
                jac_out = layer.jacobian(input, jac_out)
            else:
                raise Exception(f"unable to compute jacobian for module {module}")
            return jac_out
        
        shape_of = lambda x: x.shape if x is not None else None

        for i in range(NL):
            li    = NL - 1 - i
            layer = self.net[li]
            input = inp_list[li]
            output = inp_list[li+1]

            try:
                jac_out = get_jacobian(layer, input, output, jac_out)
            except Exception as ex:
                print(f"get_jacobian failed with module{layer} input:{shape_of(input)} jac_out:{shape_of(jac_out)}")
                raise ex

            # print(f"[{li}] layer:{layer} input:{shape_of(input)} jac_out:{shape_of(jac_out)}")
        
        jac_out_shape = list(inp_list[-1].shape) + list(inp_list[0].shape[1:])

        jac_out = jac_out.reshape(jac_out_shape)

        return jac_out