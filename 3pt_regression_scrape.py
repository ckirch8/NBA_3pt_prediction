from bs4 import BeautifulSoup, Comment
from urllib.request import urlopen
import pandas as pd
import re
import urllib.request, urllib.error
import csv
import sys


def main(argvs):

    if(argvs[1] == 'train'):
        years = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
    elif(argvs[1] == 'test'):
        years = ['2019', '2020']
    else:
        print("second arg must be 'test' or 'train'\n")
        return 0

    filename = argvs[0]
    print(argvs[0])
    base = 'https://www.basketball-reference.com'
    features_wanted = ['fg3', 'fg3a', 'fg3_pct']
    fields = ['player', 'cfg3_per_g', 'cfg3a_per_g', 'cfg3', 'cfg3a', 'cfg3_pct', 'cft', 'cfta', 'cft_pct', 'cts_pct', 'cefg_pct', 'cfg3a_per_fga_pct', 'cusg_pct', 'fg3', 'fg3a', 'fg3_pct']

    print_rows = []

    for year in years:
        print(year)
        url = base + '/leagues/NBA_' + year + '_rookies.html'
        page = urlopen(url).read()
        soup = BeautifulSoup(page, 'html.parser')
        table = soup.find("tbody")
        rows = table.find_all('tr')
        for r in rows:
            if (r.find('th', {"scope":"row"}) != None):
                name = r.find("td",{"data-stat": 'player'})
                a = name.text.strip().encode()
                name=a.decode("utf-8")

                a = r.find('a', href=True)
                p_link = a['href'].strip()
                p_url = base + p_link

                p_page = urlopen(p_url).read()
                p_soup = BeautifulSoup(p_page, 'html.parser')
                if (p_soup.find(href=re.compile('cbb/players'))):
                    row = []
                    row.append(name)
                    p_table = p_soup.find(href=re.compile('cbb/players'))
                    c_link = p_table['href'].strip()

                    try:
                        cbb_page = urlopen(c_link).read()
                    except urllib.error.HTTPError as e:
                        continue
                    cbb_soup = BeautifulSoup(cbb_page, 'html.parser')
                    foot = cbb_soup.select_one('tfoot')
                    cell = foot.find("td",{"data-stat": 'fg3_per_g'})
                    a = cell.text.strip().encode()
                    text=a.decode("utf-8")

                    row.append(text)
                    cell = foot.find("td",{"data-stat": 'fg3a_per_g'})
                    a = cell.text.strip().encode()
                    text=a.decode("utf-8")

                    row.append(text)
                    cbb_features = ['usg_pct', 'fg3a_per_fga_pct', 'efg_pct', 'ts_pct', 'ft_pct', 'fta', 'ft', 'fg3_pct', 'fg3a', 'fg3']
                    for comment in cbb_soup.find_all(string=lambda text:isinstance(text,Comment)):
                        data = BeautifulSoup(comment, 'html.parser')
                        for items in data.select("tfoot"):

                            for f in reversed(cbb_features):
                                if (items.find("td",{"data-stat": f})):
                                    cell = items.find("td",{"data-stat": f})
                                    a = cell.text.strip().encode()
                                    text=a.decode("utf-8")
                                    # print(text)
                                    row.append(text)
                                    cbb_features.remove(f)
                else:
                    continue
                for f in features_wanted:
                    cell = r.find("td",{"data-stat": f})
                    a = cell.text.strip().encode()
                    text=a.decode("utf-8")

                    row.append(text)
                print_rows.append(row)


    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(print_rows)

if __name__ == "__main__":
	main(sys.argv[1:])
