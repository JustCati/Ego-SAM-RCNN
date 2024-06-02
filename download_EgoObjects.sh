# https://ai.meta.com/datasets/egoobjects-downloads/

mkdir data/EgoObjects
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An8hVtaVFSLA4yMZFPktRgsXzMN0lbpzHWAXmD3nHmtOt0pV9u5aUW2XbTTDB2w4MgEFSWAjPz34t0chIVdMaGXDIBZ2xPGqicVHKcd1wMqEy76lMac.zip?ccb=10-5&oh=00_AYAFEw6uS55Hyqbh0OwZ4WBPXilvCLY74Mr1gDy4b492Dg&oe=66840FFE&_nc_sid=a7aa5b" --output data/EgoObjects/images.zip
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An-WS2mQvnrkM05xVRmd4NwzvUG42KxJV294Caeos-c0h8-XkxRyU9m4AdDvW5x9Sgxi4xHcXHkVkk0JyKtRZCmwCyw04Z-0ulrwQNAayOqnMvDkJvhL3nKJgtcUrA.json?ccb=10-5&oh=00_AYDPpBQU_ck9DMPBuMG4OMYaCaKPLoTMweH0ncU6Kw_CPw&oe=6684027C&_nc_sid=a7aa5b" --output data/EgoObjects/ego_objects_train.json
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An8ggk-BJQsp9pd3ra7o4f-xVlvsiNOzF7zrMHk124kuRtX_q5k3bMeO5t0LnG3LEEJuHLKZhKOYjQj7WB4dVnOtkTBG5cV4_9E4vv1KznH6Mt9SXAaTjbzJKrs.json?ccb=10-5&oh=00_AYBcgER4lS6-VxphC3CDPyjdePgLEt_WgDLe4P3pW75x0Q&oe=6683F7C5&_nc_sid=a7aa5b" --output data/EgoObjects/ego_objects_eval.json
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An8K4G08lXqX2Om6ZxT8yc0w9oEoqNjimpfZSGFLENsvJ3xB4nuKak0A762P82rRnwptKSXdgwHQm1cdHgKqRu2tTsutxrPfiz_kApnl3AmOSQNiU2njLSlnjxlI.json?ccb=10-5&oh=00_AYCNrZvpbg-dSs9l7pk0byfvaKpPVeGU8w7pa0FuF-SJgA&oe=668414BB&_nc_sid=a7aa5b" --output data/EgoObjects/ego_objects_metadata.json

unzip data/EgoObjects/images.zip -d data/EgoObjects
rm data/EgoObjects/images.zip
