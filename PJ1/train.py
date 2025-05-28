import numpy as np
from model import NeuralNetwork, LinearLayer

# =====LinearLR=====
class Trainer:
    def __init__(self, model, lr=0.01, reg=0.0, step_size=5, gamma=0.1, patience=5, momentum=0.9):
        self.model = model
        self.lr = lr
        self.reg = reg
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience

        
        # 训练指标存储
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.lr_history = []
        
        # 早停相关变量
        self.best_val_acc = 0
        self.counter = 0    # 无提升计数
        self.early_stop = False

        # 学习率线性衰减相关参数
        self.start_lr = lr  # 初始学习率
        self.end_lr = lr * gamma  # 最终学习率

        self.optimizer = SGDMomentum(
                layers=model.layers,  # 包含所有层的列表
                lr=lr,
                momentum=momentum,
                weight_decay=reg
            )
        
    def compute_accuracy(self, logits, y):
        preds = np.argmax(logits, axis=1)
        return np.mean(preds == y)



    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, verbose=True):
        for epoch in range(epochs):
            if self.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
            # === 学习率衰减（LinearLR） ===
            # 计算当前进度，不超过1.0
            progress = min(epoch / self.step_size, 1.0)
            self.lr = self.start_lr * (1 - progress) + self.end_lr * progress
            # 更新优化器的学习率
            self.optimizer.lr = self.lr

            # === 训练集迭代 ===
            train_loss, train_correct = 0.0, 0
            indices = np.random.permutation(len(X_train))
            
            for i in range(0, len(X_train), batch_size):
                # 获取当前batch
                batch_idx = indices[i:i+batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # 前向传播
                logits = self.model.forward(X_batch)
                loss = self.model.loss.forward(logits, y_batch) + self.model.get_regularization_loss(self.reg)
                
                # 反向传播与参数更新
                self.model.backward()
                self.optimizer.step()

                # 累计统计量
                train_loss += loss * len(X_batch)
                train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

            # === 计算指标 ===
            train_loss /= len(X_train)
            train_acc = train_correct / len(X_train)
            val_logits = self.model.forward(X_val)
            val_loss = self.model.loss.forward(val_logits, y_val) + self.model.get_regularization_loss(self.reg)
            val_acc = self.compute_accuracy(val_logits, y_val)
            
            # === 存储指标 ===
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.lr_history.append(self.lr)

            # === 早停判断 ===
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.counter = 0  # 重置计数器
                self.model.save_weights("best_model_medium3.npz")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

            # === 打印日志 ===
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                      f"LR: {self.lr:.6f}")

        # 训练正常结束（未触发早停）
        if not self.early_stop:
            print(f"Training completed all {epochs} epochs")



class SGDMomentum:
    def __init__(self, layers, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        

        # 关键修正：只处理LinearLayer类型层
        self.linear_layers = [layer for layer in layers if isinstance(layer, LinearLayer)]
        
        # 初始化速度缓冲区（仅针对线性层）
        self.velocities = []
        for layer in self.linear_layers:
            self.velocities.append({
                'W': np.zeros_like(layer.W),
                'b': np.zeros_like(layer.b)
            })


    def step(self):
        layer_idx = 0
        for i, layer in enumerate(self.linear_layers):
            if not isinstance(layer, LinearLayer):
                continue
                
            # ===== 权重更新 =====
            # 计算正则化梯度项
            reg_grad_W = self.weight_decay * layer.W
            
            # 更新W的速度
            v_W = self.momentum * self.velocities[layer_idx]['W'] 
            v_W += self.lr * (layer.dW + reg_grad_W)
            
            # 更新b的速度（不加正则化）
            v_b = self.momentum * self.velocities[layer_idx]['b']
            v_b += self.lr * layer.db
            
            # 应用更新
            layer.W -= v_W
            layer.b -= v_b
            
            # 保存当前速度
            self.velocities[layer_idx]['W'] = v_W
            self.velocities[layer_idx]['b'] = v_b
            
            layer_idx += 1




# # ====StepLR====
# class Trainer:
#     def __init__(self, model, lr=0.01, reg=0.0, step_size=5, gamma=0.1, 
#                  patience=5, momentum=0.9):

#         self.model = model
#         self.lr = lr
#         self.reg = reg
#         self.step_size = step_size
#         self.gamma = gamma
#         self.patience = patience
        
#         # 训练指标存储
#         self.train_losses = []
#         self.train_accs = []
#         self.val_losses = []
#         self.val_accs = []
#         self.lr_history = []
        
#         # 早停相关变量
#         self.best_val_acc = 0
#         self.counter = 0    # 无提升计数
#         self.early_stop = False

#         self.optimizer = SGDMomentum(
#                 layers=model.layers,  # 包含所有层的列表
#                 lr=lr,
#                 momentum=momentum,
#                 weight_decay=reg
#             )
        
#     def compute_accuracy(self, logits, y):
#         preds = np.argmax(logits, axis=1)
#         return np.mean(preds == y)

#     def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=True):
#         for epoch in range(epochs):
#             if self.early_stop:
#                 print(f"Early stopping triggered at epoch {epoch+1}")
#                 break
                
#             # === 学习率衰减 ===
#             if (epoch + 1) % self.step_size == 0:
#                 self.lr *= self.gamma
#                 if verbose:
#                     print(f"Epoch {epoch+1}: Learning rate decayed to {self.lr:.6f}")

#             # === 训练集迭代 ===
#             train_loss, train_correct = 0.0, 0
#             indices = np.random.permutation(len(X_train))
            
#             for i in range(0, len(X_train), batch_size):
#                 # 获取当前batch
#                 batch_idx = indices[i:i+batch_size]
#                 X_batch = X_train[batch_idx]
#                 y_batch = y_train[batch_idx]

#                 # 前向传播
#                 logits = self.model.forward(X_batch)
#                 loss = self.model.loss.forward(logits, y_batch) + self.model.get_regularization_loss(self.reg)
                
#                 # 反向传播与参数更新
#                 self.model.backward()
#                 # ====SGD====
#                 # for layer in self.model.layers:
#                 #     if isinstance(layer, LinearLayer):
#                 #         layer.W -= self.lr * (layer.dW + self.reg * layer.W)
#                 #         layer.b -= self.lr * layer.db

#                 #====SGD动量====
#                 self.optimizer.step()

#                 # 累计统计量
#                 train_loss += loss * len(X_batch)
#                 train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

#             # === 计算指标 ===
#             train_loss /= len(X_train)
#             train_acc = train_correct / len(X_train)
#             val_logits = self.model.forward(X_val)
#             val_loss = self.model.loss.forward(val_logits, y_val) + self.model.get_regularization_loss(self.reg)
#             val_acc = self.compute_accuracy(val_logits, y_val)
            
#             # === 存储指标 ===
#             self.train_losses.append(train_loss)
#             self.train_accs.append(train_acc)
#             self.val_losses.append(val_loss)
#             self.val_accs.append(val_acc)
#             self.lr_history.append(self.lr)

#             # === 早停判断 ===
#             if val_acc > self.best_val_acc:
#                 self.best_val_acc = val_acc
#                 self.counter = 0  # 重置计数器
#                 # =====是否存储最优参数=====
#                 self.model.save_weights("best_model_steplr.npz")
#             else:
#                 self.counter += 1
#                 if self.counter >= self.patience:
#                     self.early_stop = True

#             # === 打印日志 ===
#             if verbose:
#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
#                       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
#                       f"LR: {self.lr:.6f}")

#         # 训练正常结束（未触发早停）
#         if not self.early_stop:
#             print(f"Training completed all {epochs} epochs")


# # =====CosineLR=====
# class Trainer:
#     def __init__(self, model, lr=0.01, reg=0.0, patience=5, momentum=0.9, eta_min=0.0):
#         self.model = model
#         self.lr = lr
#         self.initial_lr = lr  # 新增：保存初始学习率用于余弦退火
#         self.reg = reg
#         self.patience = patience
#         self.eta_min = eta_min
        
#         # 训练指标存储
#         self.train_losses = []
#         self.train_accs = []
#         self.val_losses = []
#         self.val_accs = []
#         self.lr_history = []  # 明确记录每个epoch的学习率
        
#         # 早停相关变量
#         self.best_val_acc = 0
#         self.counter = 0
#         self.early_stop = False

#         self.optimizer = SGDMomentum(
#             layers=model.layers,
#             lr=lr,
#             momentum=momentum,
#             weight_decay=reg
#         )

#     def compute_accuracy(self, logits, y):
#         preds = np.argmax(logits, axis=1)
#         return np.mean(preds == y)

#     def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=True):
#         for epoch in range(epochs):
#             if self.early_stop:
#                 print(f"Early stopping triggered at epoch {epoch+1}")
#                 break

#             # === Cosine学习率衰减 ===
#             progress = epoch / max(epochs-1, 1)  # 处理epochs=1的情况
#             current_lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1 + np.cos(np.pi * progress))
#             self.lr = current_lr
#             self.optimizer.lr = current_lr  # 同步更新优化器参数
#             self.lr_history.append(current_lr)  # 记录学习率变化

#             # === 训练集迭代 ===
#             train_loss, train_correct = 0.0, 0
#             indices = np.random.permutation(len(X_train))
            
#             for i in range(0, len(X_train), batch_size):
#                 batch_idx = indices[i:i+batch_size]
#                 X_batch = X_train[batch_idx]
#                 y_batch = y_train[batch_idx]

#                 # 前向传播
#                 logits = self.model.forward(X_batch)
#                 loss = self.model.loss.forward(logits, y_batch) + self.model.get_regularization_loss(self.reg)
                
#                 # 反向传播与参数更新
#                 self.model.backward()
#                 self.optimizer.step()

#                 # 累计统计量
#                 train_loss += loss * len(X_batch)
#                 train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

#             # === 指标计算与存储 ===
#             train_loss /= len(X_train)
#             train_acc = train_correct / len(X_train)
#             val_logits = self.model.forward(X_val)
#             val_loss = self.model.loss.forward(val_logits, y_val) + self.model.get_regularization_loss(self.reg)
#             val_acc = self.compute_accuracy(val_logits, y_val)
            
#             self.train_losses.append(train_loss)
#             self.train_accs.append(train_acc)
#             self.val_losses.append(val_loss)
#             self.val_accs.append(val_acc)

#             # === 早停判断 ===
#             if val_acc > self.best_val_acc:
#                 self.best_val_acc = val_acc
#                 self.counter = 0
#                 # =====是否存储最优参数=====
#                 self.model.save_weights("best_model_coslr.npz")
#             else:
#                 self.counter += 1
#                 if self.counter >= self.patience:
#                     self.early_stop = True

#             # === 日志输出 ===
#             if verbose:
#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
#                       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
#                       f"LR: {current_lr:.6f}")

#         if not self.early_stop:
#             print(f"Training completed all {epochs} epochs")



# # =====LinearLR=====
# # 数据增强
# class Trainer:
#     def __init__(self, model, lr=0.01, reg=0.0, step_size=5, gamma=0.1, patience=5, momentum=0.9, augment=True, crop_padding=4, flip_prob=0.3):
#         self.model = model
#         self.lr = lr
#         self.reg = reg
#         self.step_size = step_size
#         self.gamma = gamma
#         self.patience = patience

#         # 数据增强参数
#         self.augment = augment            # 是否启用增强
#         self.crop_padding = crop_padding  # 裁剪填充像素数
#         self.flip_prob = flip_prob        # 水平翻转概率
        
#         # 训练指标存储
#         self.train_losses = []
#         self.train_accs = []
#         self.val_losses = []
#         self.val_accs = []
#         self.lr_history = []
        
#         # 早停相关变量
#         self.best_val_acc = 0
#         self.counter = 0    # 无提升计数
#         self.early_stop = False

#         # 学习率线性衰减相关参数
#         self.start_lr = lr  # 初始学习率
#         self.end_lr = lr * gamma  # 最终学习率

#         self.optimizer = SGDMomentum(
#                 layers=model.layers,  # 包含所有层的列表
#                 lr=lr,
#                 momentum=momentum,
#                 weight_decay=reg
#             )
        
#     def compute_accuracy(self, logits, y):
#         preds = np.argmax(logits, axis=1)
#         return np.mean(preds == y)

#     # 数据增强
#     def augment_batch(self, X_batch):
#         """
#         对单个batch进行在线数据增强
#         输入: X_batch形状为(batch_size, 3072)
#         输出: 增强后的X_batch形状保持(batch_size, 3072)
#         """
#         if not self.augment:
#             return X_batch

#         # 将展平数据转换为图像格式 (batch_size, 32, 32, 3)
#         X_images = X_batch.reshape(-1, 32, 32, 3)
#         augmented = []
        
#         for img in X_images:
#             # 随机水平翻转
#             if np.random.rand() < self.flip_prob:
#                 img = np.fliplr(img)

#             # 随机裁剪（仅在启用padding时执行）
#             if self.crop_padding > 0:
#                 # 给图像添加padding
#                 padded = np.pad(img, [(self.crop_padding, self.crop_padding),
#                                       (self.crop_padding, self.crop_padding), (0,0)], mode='constant')
#                 # 随机选择裁剪位置
#                 h, w = img.shape[:2]
#                 offset_h = np.random.randint(0, 2*self.crop_padding)
#                 offset_w = np.random.randint(0, 2*self.crop_padding)
#                 img = padded[offset_h:offset_h+h, offset_w:offset_w+w]

#             augmented.append(img.reshape(-1))  # 重新展平为3072维

#         return np.array(augmented)
    


#     def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, verbose=True):
#         for epoch in range(epochs):
#             if self.early_stop:
#                 print(f"Early stopping triggered at epoch {epoch+1}")
#                 break
                
#             # === 学习率衰减（LinearLR） ===
#             # 计算当前进度，不超过1.0
#             progress = min(epoch / self.step_size, 1.0)
#             self.lr = self.start_lr * (1 - progress) + self.end_lr * progress
#             # 更新优化器的学习率
#             self.optimizer.lr = self.lr

#             # === 训练集迭代 ===
#             train_loss, train_correct = 0.0, 0
#             indices = np.random.permutation(len(X_train))
            
#             for i in range(0, len(X_train), batch_size):
#                 # 获取当前batch
#                 batch_idx = indices[i:i+batch_size]
#                 X_batch = X_train[batch_idx]
#                 y_batch = y_train[batch_idx]

#                 # === 应用数据增强 ===
#                 X_batch = self.augment_batch(X_batch)

#                 # 前向传播
#                 logits = self.model.forward(X_batch)
#                 loss = self.model.loss.forward(logits, y_batch) + self.model.get_regularization_loss(self.reg)
                
#                 # 反向传播与参数更新
#                 self.model.backward()
#                 self.optimizer.step()

#                 # 累计统计量
#                 train_loss += loss * len(X_batch)
#                 train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

#             # === 计算指标 ===
#             train_loss /= len(X_train)
#             train_acc = train_correct / len(X_train)
#             val_logits = self.model.forward(X_val)
#             val_loss = self.model.loss.forward(val_logits, y_val) + self.model.get_regularization_loss(self.reg)
#             val_acc = self.compute_accuracy(val_logits, y_val)
            
#             # === 存储指标 ===
#             self.train_losses.append(train_loss)
#             self.train_accs.append(train_acc)
#             self.val_losses.append(val_loss)
#             self.val_accs.append(val_acc)
#             self.lr_history.append(self.lr)

#             # === 早停判断 ===
#             if val_acc > self.best_val_acc:
#                 self.best_val_acc = val_acc
#                 self.counter = 0  # 重置计数器
#                 self.model.save_weights("best_model_medium_with_augment.npz")
#             else:
#                 self.counter += 1
#                 if self.counter >= self.patience:
#                     self.early_stop = True

#             # === 打印日志 ===
#             if verbose:
#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
#                       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
#                       f"LR: {self.lr:.6f}")

#         # 训练正常结束（未触发早停）
#         if not self.early_stop:
#             print(f"Training completed all {epochs} epochs")





# # =====ReduceLROnPlateau=====
# class Trainer:
#     def __init__(self, model, lr=0.01, reg=0.0, gamma=0.1, lr_patience=3, lr_delta=0.001,
#                  early_stop_patience=5, momentum=0.9):

#         self.model = model
#         self.lr = lr
#         self.reg = reg
#         self.gamma = gamma
#         self.stop_patience = early_stop_patience
#         self.lr_patience = lr_patience
#         self.lr_delta = lr_delta
        
#         # 训练指标存储
#         self.train_losses = []
#         self.train_accs = []
#         self.val_losses = []
#         self.val_accs = []

#         self.lr_history = []
    
#         # 早停相关变量
#         self.best_val_acc = 0
#         self.early_stop_counter = 0    # 无提升计数
#         self.early_stop = False

#         # 学习率下降相关变量
#         self.lr_counter = 0    # 无提升计数

#         self.optimizer = SGDMomentum(
#             layers=model.layers,  # 包含所有层的列表
#             lr=lr,
#             momentum=momentum,
#             weight_decay=reg
#         )

#     def compute_accuracy(self, logits, y):
#         preds = np.argmax(logits, axis=1)
#         return np.mean(preds == y)

#     def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, verbose=True):
#         for epoch in range(epochs):
#             if self.early_stop:
#                 print(f"Early stopping triggered at epoch {epoch+1}")
#                 break
                

#             # === 训练集迭代 ===
#             train_loss, train_correct = 0.0, 0
#             indices = np.random.permutation(len(X_train))
            
#             for i in range(0, len(X_train), batch_size):
#                 # 获取当前batch
#                 batch_idx = indices[i:i+batch_size]
#                 X_batch = X_train[batch_idx]
#                 y_batch = y_train[batch_idx]

#                 # 前向传播
#                 logits = self.model.forward(X_batch)
#                 loss = self.model.loss.forward(logits, y_batch) + self.model.get_regularization_loss(self.reg)
                
#                 # 反向传播与参数更新
#                 self.model.backward()
#                 # ===SGD===
#                 # for layer in self.model.layers:
#                 #     if isinstance(layer, LinearLayer):
#                 #         layer.W -= self.lr * (layer.dW + self.reg * layer.W)
#                 #         layer.b -= self.lr * layer.db

#                 # ===SGD动量===
#                 self.optimizer.step()

#                 # 累计统计量
#                 train_loss += loss * len(X_batch)
#                 train_correct += np.sum(np.argmax(logits, axis=1) == y_batch)

#             # === 计算指标 ===
#             train_loss /= len(X_train)
#             train_acc = train_correct / len(X_train)
#             val_logits = self.model.forward(X_val)
#             val_loss = self.model.loss.forward(val_logits, y_val) + self.model.get_regularization_loss(self.reg)
#             val_acc = self.compute_accuracy(val_logits, y_val)
            
#             # === 存储指标 ===
#             self.train_losses.append(train_loss)
#             self.train_accs.append(train_acc)
#             self.val_losses.append(val_loss)
#             self.val_accs.append(val_acc)
#             self.lr_history.append(self.lr)

#             # === 学习率调整逻辑 ===
#             # === 早停判断 ===
#             if val_acc > self.best_val_acc + self.lr_delta:
#                 self.best_val_acc = val_acc
#                 self.lr_counter = 0  # 重置计数器
#                 self.early_stop_counter = 0  # 重置计数器
#                 # self.model.save_weights("best_model.npz")
#             else:
#                 self.lr_counter += 1
#                 self.early_stop_counter += 1
#                 if verbose:
#                     print(f"Validation accuracy not improved. Learning rate patience ({self.lr_counter}/{self.lr_patience})")
#                     print(f"Early stop patience ({self.early_stop_counter}/{self.stop_patience})")
                
#                 # 触发学习率下降
#                 if self.lr_counter >= self.lr_patience:
#                     old_lr = self.lr
#                     self.lr *= self.gamma
#                     self.lr_counter = 0  # 重置计数器
#                     if verbose:
#                         print(f"Learning rate reduced from {old_lr:.6f} to {self.lr:.6f}")

#                 # 触发早停
#                 if self.early_stop_counter >= self.stop_patience:
#                     self.early_stop = True


#             # === 打印日志 ===
#             if verbose:
#                 print(f"Epoch {epoch+1}/{epochs} | "
#                       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
#                       f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
#                       f"LR: {self.lr:.6f}")

#         # 训练正常结束（未触发早停）
#         if not self.early_stop:
#             print(f"Training completed all {epochs} epochs")


