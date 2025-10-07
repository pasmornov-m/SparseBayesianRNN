import torch
import time


class LMTrainer():
    def __init__(self, 
                 model,
                 optimizer,
                 criterion,
                 num_epochs,
                 train_loader,
                 valid_loader=None,
                 test_loader=None,
                 grad_clip=10.0,
                 print_fq=1,
                 save_fq=0,
                 file_name="model_params/model.pt",
                 log_interval=100):
        
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.print_fq = print_fq
        self.save_fq = save_fq
        self.file_name = file_name
        self.sparsification_eval_fun=model.evaluate_compression if hasattr(model, "evaluate_compression") else None
        self.log_interval = log_interval
        
        self.train_size = train_loader.train_size()
        self.num_batches = len(train_loader)
        
    def evaluate(self, loader):
        self.model.eval()
        loss_sum = 0.0
        tokens = 0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                base_loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                reg = self.get_reg()
                loss = base_loss + reg
                loss_sum += loss.item() * yb.numel()
                tokens += yb.numel()
        self.model.train()
        return (loss_sum / tokens) if tokens > 0 else None
    
    def get_reg(self):
        reg = sum(
            [layer.eval_reg(self.train_size) 
            for layer in self.model.modules() 
            if hasattr(layer, "eval_reg")])
        return reg
    
    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()
            total_loss_weighted = 0.0
            total_tokens = 0
            grad_norm = 0.0

            for batch_idx, (xb, yb) in enumerate(self.train_loader):

                self.optimizer.zero_grad()           
                logits = self.model(xb)
                base_loss = self.criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                reg = self.get_reg()
                loss = base_loss + reg
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                loss_item = loss.item()
                n_tokens = yb.numel()
                total_loss_weighted += loss_item * n_tokens
                total_tokens += n_tokens

                grad_norm = max(grad_norm, torch.norm(
                    torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])
                ).item())

                if (batch_idx % self.log_interval == 0) or (batch_idx == self.num_batches - 1):
                    print(
                        f"epoch {epoch:04d}/{self.num_epochs} | batch {batch_idx:04d}/{self.num_batches} | "
                        f"CE {base_loss.item():.4f} | total_loss {loss_item:.4f} | tokens {n_tokens} "
                    )

            avg_train_loss = total_loss_weighted / total_tokens if total_tokens > 0 else float("nan")
            epoch_time = time.perf_counter() - epoch_start

            self.print_epoch_metrics(epoch, epoch_time, avg_train_loss, grad_norm)

            if self.sparsification_eval_fun is not None:
                self.sparsification_eval_fun()
    
    def print_epoch_metrics(self, epoch, epoch_time, train_loss, grad_norm):
        do_print = (epoch % self.print_fq == 0) or (epoch == self.num_epochs)
        if do_print:
            val_loss = self.evaluate(self.valid_loader) if self.valid_loader is not None else None
            test_loss = self.evaluate(self.test_loader) if self.test_loader is not None else None

            val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            test_str = f"{test_loss:.4f}" if test_loss is not None else "n/a"

            print(
                f"Epoch {epoch:03d} | Time {epoch_time:.2f}s | Train {train_loss:.4f} | "
                f"grad_norm {grad_norm:.4f} | Val {val_str} | Test {test_str}"
            )
