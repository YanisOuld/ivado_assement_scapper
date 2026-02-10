WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_most-visited_museums"

MIN_VISITORS = 2_000_000

RAW_MUSEUMS_CSV = "data/raw/museums_raw.csv"
JOINED_MUSEUMS_POPULATION_CSV = "data/processed/museums_joined.csv"

POPULATION_RAW_CSV = "data/raw/population.csv"
POPULATION_PROCESSED_CSV = "data/processed/population.csv"

USER_AGENT_HEADERS =  {
  	"User-Agent": (
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
		"AppleWebKit/537.36 (KHTML, like Gecko) "
		"Chrome/120.0.0.0 Safari/537.36"
	)
}