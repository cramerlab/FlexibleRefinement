/*bool MetaDataTable::readStarList(std::ifstream& in, std::vector<EMDLabel> *desiredLabels)
{
	setIsList(true);
	addObject();
	long int objectID = objects.size() - 1;

	std::string line, firstword, value;
	std::vector<std::string> words;

	bool also_has_loop = false;

	// Read data and fill structures accordingly
	while (getline(in, line, '\n'))
	{
		tokenize(line, words);

		// Ignore empty lines
		if (words.size() == 0)
			continue;
		else
			firstword = words[0];

		// Get label-value pairs
		if (firstword[0] == '_')
		{
			EMDLabel label = EMDL::str2Label(firstword.substr(1)); // get rid of leading underscore
			if (words.size() != 2)
				REPORT_ERROR("MetaDataTable::readStarList: did not encounter a single word after " + firstword);
			value = words[1];

			if (desiredLabels != NULL && !vectorContainsLabel(*desiredLabels, label))
			{
				label = EMDL_UNDEFINED; //ignore if not present in desiredLabels
			}
			if (label != EMDL_UNDEFINED)
			{
				addLabel(label);
				setValueFromString(label, value, objectID);
			}
		}
		// Check whether there is a comment or an empty line
		else if (firstword[0] == '#' || firstword[0] == ';')
		{
			// TODO: handle comments?
			continue;
		}
		// Check whether a loop structure comes after this list
		else if (firstword.find("loop_") == 0)
		{
			also_has_loop = true;
			return also_has_loop;
		}
		// Check whether this data blocks ends (because a next one is there)
		else if (firstword.find("data_") == 0)
		{
			// Should I reverse the pointer one line?
			return also_has_loop;
		}
	}
	// Reached the end of the file
	return also_has_loop;
}

long int MetaDataTable::readStar(std::ifstream& in, const std::string &name, std::vector<EMDLabel> *desiredLabels, std::string grep_pattern, bool do_only_count)
{
	std::string line, token, value;
	clear();
	bool also_has_loop;

	// Start reading the ifstream at the top
	in.seekg(0);

	// Proceed until the next data_ or _loop statement
	// The loop statement may be necessary for data blocks that have a list AND a table inside them
	while (getline(in, line, '\n'))
	{
		// Find data_ lines
		if (line.find("data_") != std::string::npos)
		{
			token = line.substr(line.find("data_") + 5);
			// If a name has been given, only read data_thatname
			// Otherwise, just read the first data_ block
			if (name == "" || name == token)
			{
				setName(token);
				// Get the next item that starts with "_somelabel" or with "loop_"
				int current_pos = in.tellg();
				while (getline(in, line, '\n'))
				{
					trim(line);
					if (line.find("loop_") != std::string::npos)
					{
						return readStarLoop(in, desiredLabels, grep_pattern, do_only_count);
					}
					else if (line[0] == '_')
					{
						// go back one line in the ifstream
						in.seekg(current_pos);
						also_has_loop = readStarList(in, desiredLabels);
						return (also_has_loop) ? 0 : 1;
					}
				}
			}
		}
	}

	// Clear the eofbit so we can perform more actions on the stream.
	in.clear();

	return 0;
}*/