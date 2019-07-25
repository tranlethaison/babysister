cd demo
ffmpeg -i result/frames/%06d.jpg -codec:video libx264 -crf 23 result.mp4

