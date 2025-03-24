from pyteomics import mgf
import pandas as pd


def CI_search(input_mgf, output_path):
    ions = []
    # print(ions)
    with mgf.read(input_mgf) as spectra:
        for i, spectrum in enumerate(spectra):
                intensity_max = max(spectrum['intensity array'])  # intensity normalization
                intensity_min = min(spectrum['intensity array'])  # intensity normalization
                x = (spectrum['intensity array'] - intensity_min) / (intensity_max - intensity_min)  # intensity normalization
                for y, fragment_ion in enumerate(spectrum['m/z array']):
                    if x[y] >= 0.6:
                        if int(fragment_ion) not in ions:
                            ions.append(int(fragment_ion))

    print(ions)
    list_total = [ions]
    df = pd.DataFrame(data=list_total)
    df2 = pd.DataFrame(df.values.T, columns=['characteristic_ions'])
    df2.to_csv(output_path + '/Characteristic_ion.csv', encoding='gbk', index=False)
    # return ions
        #print(ions)


if __name__ == "__main__":
    input_mgf = 'C:/Users/Li/Desktop/CNNtest/test/database/Di.mgf'
    output_path = 'C:/Users/Li/Desktop/CNNtest/test/database'
    CI_search(input_mgf, output_path)
    # a = CI_search(input_mgf)
    # print(a)
