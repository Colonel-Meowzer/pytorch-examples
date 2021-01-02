#!/usr/bin/env python

import pytest
import torch
from predict import list_files, unicode_to_ascii, line_to_tensor, letter_to_tensor

@pytest.mark.parametrize("input,output",[
    ("Ślusàrski", "Slusarski")
])
def test_Given_unicodeName_When_unicodeToAscii_Then_returnAsciiName(input, output):
    assert unicode_to_ascii(input) == output

@pytest.mark.parametrize("input,output",[
    ("data/names/*.txt", ['data/names/French.txt', 'data/names/Czech.txt', 'data/names/Dutch.txt', 'data/names/Polish.txt', 'data/names/Scottish.txt', 'data/names/Chinese.txt', 'data/names/English.txt', 'data/names/Italian.txt', 'data/names/Portuguese.txt', 'data/names/Japanese.txt', 'data/names/German.txt', 'data/names/Russian.txt', 'data/names/Korean.txt', 'data/names/Arabic.txt', 'data/names/Greek.txt', 'data/names/Vietnamese.txt', 'data/names/Spanish.txt', 'data/names/Irish.txt'])
])
def test_Given_validPath_When_listFiles_Then_listAllFiles(input, output):
    assert list_files(input) == output

@pytest.mark.parametrize("input,output", [
    ("J", """tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])""")
])
def test_Given_validLetter_When_letterToTensor_Then_returnTensor(input, output):
   assert letter_to_tensor(input) == output 



@pytest.mark.parametrize("input,output", [
    ("Jones", torch.Size([5, 1, 57]))
])
def test_Given_validLine_When_lineToTensor_Then_returnTensorSize(input, output):
   assert line_to_tensor(input).size() == output 
