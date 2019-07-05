# #!/Users/gnebie/.brew/bin/


import argparse
import linearRegression as linearRegression
import json



def get_settings():
	with open('linear_regeression.conf') as json_file:
		data = json.load(json_file)
	return data


def parse_program_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("file_name", help="File path of the wiki object you find from https://dumps.wikimedia.org/frwiki/latest/")
	parser.add_argument("type", type=int, default=0, help="type")
	return parser.parse_args()

def create_model(lr, args):
		print("lr.get_values_from_file(args.file_name, args.type)")
		lr.get_values_from_file(args.file_name, args.type)
		print("lr.improve_data()")
		lr.improve_data()
		print("lr.calculate()")
		lr.calculate()
		print("lr.print_infos()")
		lr.print_infos()
		print("lr.save_model()")
		lr.save_model()

def get_model(lr, args):
		lr.get_values_from_file(args.file_name, args.type)
		lr.load_model("test_0")
		lr.improve_data()
		# lr.print_infos()
		lr.print_wait_graphe()


def main():
	try:
		args = parse_program_args()

		args.file_name = "~/goinfre/download/calcofi/parsed_bottle.csv"
		args.file_name = "~/goinfre/download/calcofi/parsed_bottle2.csv"
		args.file_name = "~/goinfre/download/calcofi/tmp_small_bottle.csv"
		settings = {}
		try:
			settings = get_settings()
		except IOError:
			print("Error when setting try to be open")
		lr = linearRegression.linearRegression(settings)
		create_model(lr, args)
		# get_model(lr, args)

	except IOError:
		print("Error when file tryed to be open")
		exit(0)

if __name__ == '__main__':
	main()
