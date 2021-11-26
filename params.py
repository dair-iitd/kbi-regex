from regexkb import constants


def add_args(parser):

    # Dataset
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=False, default='fb15k')

    # Model
    parser.add_argument(
        '-m', '--model', help="model name as in models.py", required=False, default='RotatE')
    parser.add_argument('-dim', '--embedding_dim', help="embedding dimension of entities and relations",
                        required=False, type=int, default=400)
    parser.add_argument('--box', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug2', action='store_true')
    parser.add_argument('--truncate_loss', action='store_true')
    # parser.add_argument('-sub_dim', '--sub_embedding_dim', help="sub_embedding dimension",
    #                     required=False, type=int, default=5)
    # parser.add_argument('--offset_type', required=False, default='abs')
    # parser.add_argument('--norm', required=False, type=int, default=1)
    parser.add_argument('--alpha', required=False, type=float, default=0.2)
    # parser.add_argument('--init', required=False, default='normal')
    # parser.add_argument('--lambd', required=False, type=float, default=4.0)
    # parser.add_argument('--load_rotate', required=False)
    # parser.add_argument('--pos_embd', action='store_true')
    # parser.add_argument('--pos_embd_threshold',
    #                     required=False, type=int, default=0)
    # parser.add_argument('--rel_path_add_unk', action='store_true')
    kleene_plus_op_choices = [constants.FREE_PARAM,
                              constants.GEOMETRIC, constants.GQE, constants.HYBRID]
    parser.add_argument('--kleene_plus_op', help=f'Kleene Plus operation, choice between {kleene_plus_op_choices}',
                        required=False, choices=kleene_plus_op_choices, default=constants.GEOMETRIC)
    parser.add_argument('--disjunction_op', help=f'Disjunction operation choice',
                        required=False, default=None)
    parser.add_argument('--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')

    # # Training
    parser.add_argument('--max_epochs', required=False, type=int, default=1500)
    parser.add_argument('--gpus', required=False, type=int, default=2)
    parser.add_argument('--num_workers', required=False, type=int, default=8)
    parser.add_argument('--optimizer', required=False, default='Adam')
    parser.add_argument('--loss', required=False, default='query2box_loss')
    parser.add_argument('-lr', '--learning_rate',
                        required=False, type=float, default=0.0001)
    # parser.add_argument('-reg', '--regularization_coefficient',
    #                     required=False, type=float, default=0.0)
    parser.add_argument('--margin',
                        required=False, type=float, default=24.0)
    parser.add_argument('--epsilon',
                        required=False, type=float, default=2.0)
    parser.add_argument('--batch_size',
                        required=False, type=int, default=1024)
    parser.add_argument('-x', '--evaluate_after_n_steps',
                        required=False, type=int, default=10000)
    parser.add_argument('--eval_batch_size',
                        required=False, type=int, default=50)
    parser.add_argument('--negative_sample_count',
                        required=False, type=int, default=256)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--query_types', type=int, nargs='+', required=False, default=[0, 1, 21, 4, 5, 12, 15],
                        help='query types on which to train and evaluate')

    parser.add_argument('-adv', '--neg_adv_sampling', action='store_true')
    parser.add_argument('-a', '--adv_temperature',
                        default=1.0, type=float)

    parser.add_argument('--lr_schedule', required=False)
    parser.add_argument('--sub_sample', action='store_true')
    parser.add_argument('--resume_from_checkpoint', required=False)

    # # Others
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save_dir',
                        required=True)

    return parser
