import torch

try:
    import strided_batched_gemm

    class StidedBMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input1, input2):
            ctx.save_for_backward(input1, input2)
            output = torch.empty((input1.size(1), input1.size(0), input2.size(2)), dtype=input1.dtype,
                                 device=torch.device('cuda')).transpose(1, 0)
            if (input1.dtype == torch.float16) and (input2.dtype == torch.float16):
                output = strided_batched_gemm.strided_batched_gemm(0.0, output, 1.0, input1, input2)
            else:
                output = torch.bmm(input1, input2, out=output)
            return output.detach()

        @staticmethod
        def backward(ctx, grad_output):
            input1, input2 = ctx.saved_tensors
            grad_input2 = torch.empty((input2.size(1), input2.size(0), input2.size(2)), dtype=input2.dtype,
                                      device=torch.device('cuda')).transpose(1, 0)
            grad_input1 = torch.empty((input1.size(0), input1.size(1), input1.size(2)), dtype=input2.dtype,
                                      device=torch.device('cuda'))
            if (grad_output.dtype == torch.float16) and (input1.dtype == torch.float16) and (
                    input2.dtype == torch.float16):
                grad_input1 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input1, 1.0, grad_output,
                                                                        input2.transpose(1, 2))
                grad_input2 = strided_batched_gemm.strided_batched_gemm(0.0, grad_input2, 1.0, input1.transpose(1, 2),
                                                                        grad_output)
            else:
                grad_input1 = torch.bmm(grad_output, input2.transpose(1, 2))
                grad_input2 = torch.bmm(input1.transpose(1, 2), grad_output, out=grad_input2)
            return grad_input1, grad_input2
    batched_mm = StidedBMM.apply
except:
    batched_mm = torch.bmm