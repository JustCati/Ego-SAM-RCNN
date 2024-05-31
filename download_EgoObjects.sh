# https://ai.meta.com/datasets/egoobjects-downloads/

mkdir data/EgoObjects
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An_Xr6bYHFk8LD7WgVukX5Ih9vH0rx4vlzdIIfHWvi3ovp5Qdm3OsGzQkuKWmKIrmL4jC_e6pGXMSz8h3b34Kt_WU5-iOH2uzw.zip?ccb=10-5&oh=00_AYDlChzlAWFOX25JSfvMtf_H7h9_epejLVr3hW0vjMYxfg&oe=66815AB2&_nc_sid=a7aa5b" --output data/EgoObjects/images.zip
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An-FYElL2gu20TnBC9-nm_tGHmh6ZZRONpcELNFOg9J3Qczj5F7L5G-VyOVYmQglp1VCjHrJpdezUTDck24zNv2t54jGq4MoewHje0nvQz81xvV6upfvVRdVXOw.json?ccb=10-5&oh=00_AYAoR9QeMZJqG81WIImCOf-svUEke8sXyz0GkTrKyzv4Qg&oe=66813AB2&_nc_sid=a7aa5b" --output data/EgoObjects/ego_objects_challenge_train.json
curl "https://scontent.fcta2-1.fna.fbcdn.net/m1/v/t6/An_K6faEdgSyCx7Cs0g5d9TnjDA1GbfkA48gRVryzZX7e5rWVG-OuUeYHaj9LB18-v-TMc1sVo1q81psNGoPiMsLL4G7Det_sNMFwEkc-Qi9bPBi3oZwczI.json?ccb=10-5&oh=00_AYAHGqwC3jnDLLPMYIfUpPxQdluOMpw3PoFHc9OhPzqO2A&oe=66813B78&_nc_sid=a7aa5b" --output data/EgoObjects/ego_objects_challenge_test.json

unzip data/EgoObjects/images.zip -d data/EgoObjects
rm data/EgoObjects/images.zip
