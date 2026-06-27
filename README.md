# maddalenabin.github.io

Personal website of Maddalena Bin — Senior Data Scientist and WeRoad travel coordinator based in Stockholm.

Built with Jekyll and custom CSS. Hosted on GitHub Pages at [maddalenabin.github.io](https://maddalenabin.github.io).

## Structure

- **Home** — intro, social links, next WeRoad trip
- **Travels** — visited countries map, trip log, WeRoad coordinator info
- **CV** — work experience, education, skills, awards
- **Publications** — research papers grouped by year
- **Blog** — occasional writing

## Stack

- [Jekyll](https://jekyllrb.com/) — static site generator
- Custom CSS (no frameworks)
- [Inter](https://fonts.google.com/specimen/Inter) + [Lora](https://fonts.google.com/specimen/Lora) via Google Fonts
- [Datawrapper](https://www.datawrapper.de/) — visited countries map
- Hosted on GitHub Pages

## Run locally

```bash
bundle install
bundle exec jekyll serve --port 4001
```

Open [http://localhost:4001](http://localhost:4001).

## Update content

| What | Where |
|------|-------|
| Bio and intro | `index.html` |
| CV entries | `_data/cv.yml` |
| Projects | `_data/projects.yml` |
| Publications | `_data/publications.yml` |
| Trips | `travels.html` |
| Blog posts | `_posts/YYYY-MM-DD-title.md` |
| Navigation | `_layouts/default.html` |
| Styles | `assets/css/main.scss` |

## License

Content © Maddalena Bin. Code is MIT licensed.