
import pandas as pd
import csv
import re
from lyricsgenius import Genius
from mpi4py import MPI

songs = pd.read_csv('billboard.csv')
songs['year'] = pd.DatetimeIndex(songs['date']).year
songs.drop(['date', 'rank', 'last-week', 'peak-rank', 'weeks-on-board'], axis=1, inplace=True)
songs.drop_duplicates(inplace=True)
sampled_df = songs.groupby('year').sample(n=150, random_state=42, replace=False)

token = 'JyD9-_LdqpyHilUA3esIXZt0Dp4Ftcc4TMo9cq7b4q2zLVfDDR1AYtIoXsXnf1DS'
genius = Genius(token)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Remix)", "(Live)"] # Exclude songs with these words in their title

def scraper(years, file_name):
    with open(file_name+'.csv', 'w', newline="", encoding='utf-8') as csvfile:
        output = csv.writer(csvfile)
        output.writerow(['Year', 'Artist', 'Song Name', 'Lyrics'])
        for year in range(years[0], years[1]):
            year_df = sampled_df[sampled_df['year'] == year]
            for i in range(150):
                song_name = year_df.iloc[i, 0]
                song_artist = re.split('& | Featuring | And', year_df.iloc[i, 1]) + [year_df.iloc[i, 1]]
                try:
                    for j in song_artist:
                        if genius.search_artist(j , max_songs=1, sort="title"):
                            artist = genius.search_artist(j , max_songs=1, sort="title")
                            try:
                                song = artist.song(song_name)
                                lyrics = song.lyrics
                                output.writerow([year, year_df.iloc[i, 1], song_name, lyrics])
                                break
                            except:
                                continue
                        else:
                            continue
                except:
                    pass

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

base = 1958 + 4 * rank
years = (base, base + 4)
scraper(years, file_name=f'part{rank}')
print(f'part{rank} done!')
