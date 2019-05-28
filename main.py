# #!/Users/gnebie/.brew/bin/


import argparse
import linearRegression as linearRegression

def parse_program_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("file_name", help="File path of the wiki object you find from https://dumps.wikimedia.org/frwiki/latest/")
	parser.add_argument("type", type=int, default=0, help="type")
	parser.add_argument("options", type=int, default=0, help="options")
	return parser.parse_args()

def main():
	try:
		args = parse_program_args()
		lr = linearRegression.linearRegression(args.options)
		lr.get_values_from_file(args.file_name, args.type)
		lr.improve_data()
		lr.calculate()
		lr.print_infos()
		lr.save_model()
	except IOError:
		print("Error when file tryed to be open")
		exit(0)

if __name__ == '__main__':
	main()
