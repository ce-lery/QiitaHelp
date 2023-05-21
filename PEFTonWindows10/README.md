# Qiita「PEFTを使用したLoRA (Windows10（非WSL2使用） bitsandbytes==0.37.0使用)」のヘルプページ

## 使い方

以下の手順で、ご使用ください。  

1. [bitsandbytes==0.37のwindowsでの設定方法](#bitsandbytes==0.37のwindowsでの設定方法)の手順を実施  
2. 以下のコマンドを実行
   ```powershell
   pip install -r requirements.txt
   pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
   ```