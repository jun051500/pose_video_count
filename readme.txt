プログラム説明

本プログラムは腹筋トレーニングをアシストするアプリです。
目標回数の8割まではビップ音で知らせをし、
8割回数からは'頑張れ'のメッセージを送ります。
また、目標回数の到達した場合は’お疲れ様’のメッセージを送ります。
終了はESCキーを押すか、目標回数まで腹筋ができれば終了になります。


１．仮想環境作成
　　Python　：　3．8.13

２．追加パッケージをインストール
　pip install mediapipe
  pip install pygame

３．実行に必要なファイル
　pose_movie_count.py　←　実行ファイル
  training.mp4  ← 入力用ファイル
  seton.mp3　←　肩が下に正しく来た場合の音
  counter.mp3　←　正しく運動できた場合の音
  cheer2.mp3　←　’頑張れ’の応援メッセージ　
  good_job.mp3 ←　'お疲れ様'の終了メッセージ
  result_video.mp4 ←　プログラム実行結果の動画ファイル

４．実行方法
　mediapipeとpygameをインストールしたpython仮想環境下で
　>>　python  pose_movie_count.py
　を実行する



