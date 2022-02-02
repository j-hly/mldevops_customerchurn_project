import os
import logging
import churn_library
import constants
# import churn_library_solution as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		model = churn_library.Model("unittest")
		model.import_data(constants.path_rawdata)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		df = model.get_df_raw()
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_encoder_helper():
	'''
	test encoder helper
	model
	'''
	try:
		logging.info("Performing ecoder test...")
		model = churn_library.Model("unittest")
		model.import_data(constants.path_rawdata)
		df = model.get_df_raw()
		# check that df has all the columns required for econding
		assert set(constants.cat_columns).issubset(df.columns), "Some categorical columns are missing!"
		logging.info("    All categtorical columns are in!")
		model.process_df()
		logging.info("Encode success")
	except Exception as e:
		logging.error(e)
		raise e


def test_eda():
	'''
	test perform eda function
	model object must have been loaded with data and cleaned
	'''
	try:
		model = churn_library.Model("unittest")
		model.import_data(constants.path_rawdata)
		model.process_df()
		logging.info("Eda success!")
	except Exception as e:
		logging.error(e)
		raise e


def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	try:
		model = churn_library.Model("unittest")
		model.import_data(constants.path_rawdata)
		model.process_df()
		model.perform_feature_engineering()
		logging.info("Feature engineering success!")
	except Exception as e:
		logging.error(e)
		raise e

def test_train_models():
	'''
	test train_models
	'''
	try:
		model = churn_library.Model("unittest")
		model.import_data(constants.path_rawdata)
		model.process_df()
		model.perform_feature_engineering()
		model.train_models()
		logging.info("Model trained!")
	except Exception as e:
		logging.error(e)
		raise e







