import scrapy
from pathlib import Path
import pickle
from slugify import slugify


class FifaHtmlSpider(scrapy.Spider):
    name = "fifahtml"

    save_data_root = Path(__file__).parent / ".." / ".." / ".." / "data" / "fifahtml"

    # TODO - run this for extended period of time to get all players
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_data_root.mkdir(exist_ok=True)

    def start_requests(self):
        urls = [
            "https://www.fifaindex.com/players/fifa17_173/",
            "https://www.fifaindex.com/players/fifa16_73/",
            "https://www.fifaindex.com/players/fifa15_14/",
            "https://www.fifaindex.com/players/fifa14_13/",
            "https://www.fifaindex.com/players/fifa13_10/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for row in response.css("tr td"):
            link = row.css("a::attr(href)").extract()
            # print(name, link)
            if link:
                if "/player/" in link[0]:
                    url = response.urljoin(link[0])
                    yield scrapy.Request(url, callback=self.parse_player)

        next_page = None
        link_names = response.css("a.page-link::text").extract()
        links = response.css("a.page-link::attr(href)").extract()
        for name, link in zip(link_names, links):
            if name.lower() == "next page":
                next_page = link
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_player(self, response):
        season = response.request.url.split("/")[-2]

        name = slugify(response
                       .css("div.col-lg-8")
                       .css("div.card")
                       .css("h5.card-header::text")
                       .extract()[0])

        team = slugify(response.css("div.team")
                       .css("a.link-team::attr(title)")[0]
                       .extract())

        saved_path = (self.save_data_root / season / team / name).with_suffix(".html")
        saved_path.parent.mkdir(exist_ok=True, parents=True)
        if not saved_path.exists():
            saved_path.write_bytes(response.body)
            yield {
                "name": name,
                "season": season,
                "url": response.request.url,
                "path": str(saved_path),
            }


class FifaSpider(scrapy.Spider):
    name = "fifastats"

    # TODO - run this for extended period of time to get all players

    def start_requests(self):
        urls = [
            "https://www.fifaindex.com/players/fifa17_173/",
            "https://www.fifaindex.com/players/fifa16_73/",
            "https://www.fifaindex.com/players/fifa15_14/",
            "https://www.fifaindex.com/players/fifa14_13/",
            "https://www.fifaindex.com/players/fifa13_10/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for row in response.css("tr td"):
            link = row.css("a::attr(href)").extract()
            # print(name, link)
            if link:
                if "/player/" in link[0]:
                    url = response.urljoin(link[0])
                    yield scrapy.Request(url, callback=self.parse_player)

        next_page = None
        link_names = response.css("a.page-link::text").extract()
        links = response.css("a.page-link::attr(href)").extract()
        for name, link in zip(link_names, links):
            if name.lower() == "next page":
                next_page = link
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)

    @staticmethod
    def parse_player(response):
        name = (response
                .css("div.col-lg-8")
                .css("div.card")
                .css("h5.card-header::text")
                .extract()[0])

        team = (response.css("div.team")
                .css("a.link-team::attr(title)")[0]
                .extract())

        #position = response.css("div.team").css("span.float-right").css("a::attr(title)").extract()[0]
        position = response.css("div.card-body").css("span.float-right").css("a::attr(title)").extract()[0]
        number = response.css("div.team").css("span.float-right::text")[0].extract()
        rating = (response
                  .css("div.col-lg-8")
                  .css("div.card")
                  .css("h5")
                  .css("span.rating::text").extract()[0])
        nationality = slugify(response.css("a.link-nation::attr(title)").extract()[0])

        yield {
            "name": slugify(name),
            "info": {
                "raw team": team,
                "team": slugify(team),
                "position": position,
                "raw name": name,
                "rating": int(rating),
                "kit number": number,
                "nationality": nationality,
                "url": response.request.url,
            },
        }


class MatchSpider(scrapy.Spider):
    name = "matchlineups"

    # TODO - want the other names - not full names

    def start_requests(self):
        urls = [
            "http://www.betstudy.com/soccer-stats/c/france/ligue-1/d/results/2017-2018/",
            "http://www.betstudy.com/soccer-stats/c/france/ligue-1/d/results/2016-2017/",
            "http://www.betstudy.com/soccer-stats/c/france/ligue-1/d/results/2015-2016/",
            "http://www.betstudy.com/soccer-stats/c/france/ligue-1/d/results/2014-2015/",
            "http://www.betstudy.com/soccer-stats/c/france/ligue-1/d/results/2013-2014/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_fixtures_page)

    def parse_fixtures_page(self, response):
        for info_button in response.css("ul.action-list").css("a::attr(href)"):
            url = response.urljoin(info_button.extract())
            yield scrapy.Request(url, callback=self.parse_match_page)

    def parse_match_page(self, response):

        home_team, away_team = response.css("div.player h2 a::text").extract()

        date = response.css("em.date").css("span.timestamp::text").extract_first()

        url = response.request.url

        match_number = response.request.url.split("-")[-1].split("/")[0]

        home_goals, away_goals = (
            response.css("div.info strong.score::text").extract_first().split("-")
        )

        for table in response.css("div.table-holder"):
            if table.css("h2::text").extract_first() == "Lineups and subsitutes":
                lineups = table

        home_lineup_css = lineups.css("table.info-table")[0]
        away_lineup_css = lineups.css("table.info-table")[1]

        home_lineup_raw = [
            slugify(x)
            for x in home_lineup_css.css("tr td.left-align")
            .css("a::attr(title)")
            .extract()
        ]
        away_lineup_raw = [
            slugify(x)
            for x in away_lineup_css.css("tr td.left-align")
            .css("a::attr(title)")
            .extract()
        ]

        home_lineup = [
            slugify(x)
            for x in home_lineup_css.css("tr td.left-align").css("a::text").extract()
        ]
        away_lineup = [
            slugify(x)
            for x in away_lineup_css.css("tr td.left-align").css("a::text").extract()
        ]

        home_lineup_number = [
            int(x) for x in home_lineup_css.css("tr td.size23 strong::text").extract()
        ]
        away_lineup_number = [
            int(x) for x in away_lineup_css.css("tr td.size23 strong::text").extract()
        ]

        home_lineup_nationality = [
            slugify(x)
            for x in home_lineup_css.css("tr td.left-align")
            .css("img.flag-ico::attr(alt)")
            .extract()
        ]
        away_lineup_nationality = [
            slugify(x)
            for x in away_lineup_css.css("tr td.left-align")
            .css("img.flag-ico::attr(alt)")
            .extract()
        ]

        yield {
            "match number": int(match_number),
            "info": {
                "date": date,
                "home team": slugify(home_team),
                "away team": slugify(away_team),
                "home goals": int(home_goals),
                "away goals": int(away_goals),
                "home lineup raw names": home_lineup_raw,
                "away lineup raw names": away_lineup_raw,
                "home lineup names": home_lineup,
                "away lineup names": away_lineup,
                "home lineup numbers": home_lineup_number,
                "away lineup numbers": away_lineup_number,
                "home lineup nationalities": home_lineup_nationality,
                "away lineup nationalities": away_lineup_nationality,
                "url": url,
            },
        }


class FifaIndexTeamScraper(scrapy.Spider):
    name = "fifa-index-team"

    # TODO - run this for extended period of time to get all players

    def start_requests(self):
        urls = [
            "https://www.fifaindex.com/teams/",
            "https://www.fifaindex.com/teams/fifa18_278/",
            "https://www.fifaindex.com/teams/fifa17_173/",
            "https://www.fifaindex.com/teams/fifa16_73/",
            "https://www.fifaindex.com/teams/fifa15_14/",
            "https://www.fifaindex.com/teams/fifa14_13/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        links = [a.extract() for a in response.css("td a::attr(href)")]
        for link in links:
            if "/team/" in link:
                url = response.urljoin(link)
                yield scrapy.Request(url, callback=self.parse_team)

        next_page = None
        link_names = response.css("a.page-link::text").extract()
        links = response.css("a.page-link::attr(href)").extract()
        for name, link in zip(link_names, links):
            if name.lower() == "next page":
                next_page = link
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse_team)

    def parse_team(self, response):
        team = slugify(response.css("div.pl-3").css("h1::text").extract_first())

        for tr in response.css("table.table-players").css("tbody").css("tr"):
            rows = tr.css("td")

            # skip not valid rows (like loaned players who don't have kit number)
            if not rows or not len(rows) == 8:
                continue

            number = int(rows[0].css("::text").extract_first())

            position = rows[1].css("::text").extract_first()

            name = rows[2].css("a.link-player::attr(title)").extract_first()

            nationality = slugify(
                rows[3].css("a.link-nation::attr(title)").extract_first()
            )

            rating = rows[4].css("::text").extract_first()

            if position in ("Sub", "Res"):
                position = rows[6].css("a.link-position::attr(title)").extract_first()

            yield {
                "name": slugify(name),
                "team": team,
                "position": position,
                "rating": int(rating),
                "number": number,
                "nationality": nationality,
                "url": response.request.url,
            }


class FixturesSpider(scrapy.Spider):
    name = "fixtures"

    # TODO - want the other names - not full names

    def start_requests(self):
        urls = [
            "http://www.betstudy.com/soccer-stats/c/england/premier-league/d/fixtures/"
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_fixtures)

    @staticmethod
    def parse_fixtures(response):
        for fixture in response.css("tr")[1:]:
            home_team = fixture.css("td.right-align a::text").extract_first()
            away_team = fixture.css("td.left-align a::text").extract_first()
            date = fixture.css("td::text").extract_first()
            yield {
                "date": date,
                "home team": slugify(home_team),
                "away team": slugify(away_team),
                "url": response.request.url,
            }
